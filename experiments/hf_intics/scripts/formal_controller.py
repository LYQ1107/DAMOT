"""U3 gate, exact HF training, mandatory stage1 hold, then frozen evaluations."""
import concurrent.futures
import json
import os
import shutil
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
Q=Path(__file__).resolve().parents[1]
PY='/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def state(status,**kwargs):
    dump(Q/'manifests/formal_controller.json',dict(status=status,pid=os.getpid(),time=time.time(),**kwargs))

def alive(pid):
    try:return Path('/proc/%d/stat'%pid).read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def report():
    subprocess.run([sys.executable,str(Q/'scripts/report.py')],check=True)

def child(name,args,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1',
        PYTHONPATH=str(Q/'code'))
    with (Q/'logs'/(name+'.log')).open('w') as log:
        proc=subprocess.Popen(args,cwd=str(Q/'code'),env=env,stdout=log,stderr=subprocess.STDOUT)
        dump(Q/'manifests'/(name+'_inference.json'),dict(status='running',pid=proc.pid,
            gpu=gpu,command=args,own_training_paused=True,time=time.time()))
        code=proc.wait()
    dump(Q/'manifests'/(name+'_inference.json'),dict(status='complete' if not code else 'failed',
        pid=proc.pid,gpu=gpu,command=args,exit_code=code,own_training_paused=True,time=time.time()))
    if code:raise RuntimeError(name+' failed; inspect preserved log')

def idle_job(name,args):
    command=[sys.executable,str(Q/'scripts/gpu_job.py'),'--name',name,'--gpus','1',
        '--prefer','2,3,5,6,4,7,8,9','--']+args
    with (Q/'logs'/(name+'.log')).open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)

def gate():
    import csv
    job=json.loads((Q/'manifests/formal_job.json').read_text())
    assert job['status']=='running' and len(job['gpus'])==4
    waiting=json.loads((Q/'episode/stage1_gate_waiting.json').read_text())
    assert list(map(int,waiting['physical_gpus'].split(',')))==job['gpus']
    state('stage1_frozen_checks',own_paused_training_gpus=job['gpus'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[
            pool.submit(child,'gate_stage1_overcast',[PY,str(Q/'code/tools/run/evaluate_apr_complement.py'),
                '--run-root',str(Q),'--stage','1','--domain','overcast'],job['gpus'][0]),
            pool.submit(child,'gate_stage1_probe',[PY,str(Q/'scripts/stage_probe.py'),
                '--stage','1','--gpu',str(job['gpus'][1])],job['gpus'][1])]
        for result in concurrent.futures.as_completed(tasks):result.result()
    metrics=list(csv.DictReader((Q/'results/stage1/overcast/metrics.csv').open()))
    overall=next(r for r in metrics if r['category']=='OVERALL')
    p1=json.loads((Q/'probe/stage1/P1.json').read_text())
    protocol=json.loads((Q/'manifests/stage1_gate_protocol.json').read_text())
    hota=float(overall['HOTA_percent']);delta=hota-protocol['HF_R11_OVERALL_HOTA']
    reasons=[]
    if delta<=-.5:reasons.append('OVERALL HOTA dropped at least0.5')
    angle_field=protocol.get('angle_metric_field','first_direction_to_offline_span_deg')
    gate_angle=float(p1[angle_field])
    if gate_angle>45:reasons.append('bank PC1/offline FAP two-dimensional subspace principal angle exceeds45deg')
    decision=dict(status='stop' if reasons else 'pass',reasons=reasons,HOTA=hota,
        HF_R11_HOTA=protocol['HF_R11_OVERALL_HOTA'],HOTA_delta=delta,
        first_direction_angle_deg=p1['first_direction_angle_deg'],
        first_direction_to_offline_span_deg=p1['first_direction_to_offline_span_deg'],
        gate_angle_deg=gate_angle,gate_angle_field=angle_field,
        gate_angle_definition=protocol['angle_gate_definition'],
        principal_angles_deg=p1['principal_angles_deg'],
        P1_weather_shift_energy_share=p1['P1_weather_shift_energy_share'],
        P1_identity_energy_share=p1['P1_identity_energy_share'],seed=1790355234,time=time.time())
    dump(Q/'episode/stage1_gate.json',decision);report()
    state('stage1_gate_'+decision['status'],decision=decision)

def main():
    while not (Q/'manifests/preflight.json').exists():
        smoke=Q/'manifests/smoke_controller.json'
        if smoke.exists() and json.loads(smoke.read_text())['status'] in ['failed','failed_gate']:
            state('preflight_failed_no_formal_training');report();return
        state('waiting_for_U3');time.sleep(15)
    preflight=json.loads((Q/'manifests/preflight.json').read_text())
    if preflight['status']!='passed':
        state('preflight_failed_no_formal_training');report();return
    assert json.loads((Q/'preflight/source_checks.json').read_text())['status']=='passed'
    assert json.loads((Q/'preflight/input_storage_checks.json').read_text())['status']=='passed'
    assert json.loads((Q/'preflight/NAS_storage_checks.json').read_text())['status']=='passed'
    assert json.loads((Q/'manifests/NAS_migration.json').read_text())['status']=='verified'
    while shutil.disk_usage(str(Q)).free < 8*2**30:
        state('waiting_for_initial_checkpoint_space',required_free_GiB=8)
        time.sleep(15)
    state('auditing_HF_config_and_sealing_source')
    subprocess.run([PY,str(Q/'scripts/record_source.py')],check=True)
    dispatch=Q/'manifests/formal_dispatch.json'
    proc=None
    if dispatch.exists() and alive(json.loads(dispatch.read_text())['pid']):
        pid=json.loads(dispatch.read_text())['pid']
    elif (Q/'episode/train_complete.json').exists() or (Q/'episode/train_stopped.json').exists():
        pid=-1
    else:
        if (Q/'manifests/formal_job.json').exists():
            previous=json.loads((Q/'manifests/formal_job.json').read_text())
            assert previous['status']!='failed','Inspect failure and epoch recovery before relaunch'
        with socket.socket() as sock:
            sock.bind(('',0));port=sock.getsockname()[1]
        recovery=Q/'manifests/stage_boundary_recovery.json'
        entry=(Q/'scripts/train_with_resume_audit.py' if recovery.exists()
               else Q/'code/tools/run/apr_complement_experiment.py')
        args=[sys.executable,str(Q/'scripts/gpu_job.py'),'--name','formal','--gpus','4',
            '--prefer','2,3,5,6,4,7,8,9','--',PY,'-m','torch.distributed.launch',
            '--nproc_per_node=4','--master_port='+str(port),'--use_env',
            str(entry),'--run-root',str(Q),'--mode','train']
        log=(Q/'logs/formal.log').open('w')
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        pid=proc.pid;dump(dispatch,dict(pid=pid,command=args,time=time.time()))
    state('formal_training_or_idle_GPU_queue',training_job_pid=pid)
    while pid>0 and alive(pid):
        if (Q/'episode/stage1_gate_waiting.json').exists() and not (Q/'episode/stage1_gate.json').exists():
            gate()
        time.sleep(5)
    if proc is not None:assert proc.wait()==0
    job=json.loads((Q/'manifests/formal_job.json').read_text())
    assert job['status']=='complete',job
    if (Q/'episode/train_stopped.json').exists():
        report()
        subprocess.run([sys.executable,str(Q/'scripts/validate_completed.py')],check=True)
        state('stopped_by_stage1_gate_awaiting_publication');return
    assert (Q/'episode/train_complete.json').exists()
    state('final_frozen_evaluation')
    args=[]
    domains=['overcast','rainy','dawn_dusk','night']
    for stage in range(2,5):
        for domain in domains[:stage]:
            args.append(('eval_stage%d_%s'%(stage,domain),[PY,str(Q/'code/tools/run/evaluate_apr_complement.py'),
                '--run-root',str(Q),'--stage',str(stage),'--domain',domain]))
    args.append(('probe_stage4',[PY,str(Q/'scripts/stage_probe.py'),'--stage','4','--gpu','0']))
    # The GPU wrapper sets one physical card visible; stage_probe's explicit
    # gpu argument must use that same physical number, so dispatch via helper.
    args[-1]=('probe_stage4',[PY,str(Q/'scripts/probe_visible_gpu.py')])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(idle_job,name,command) for name,command in args]
        for result in concurrent.futures.as_completed(futures):result.result()
    report()
    subprocess.run([sys.executable,str(Q/'scripts/validate_completed.py')],check=True)
    state('complete_awaiting_publication')

if __name__=='__main__':
    try:main()
    except BaseException:
        state('failed',error=traceback.format_exc())
        raise
