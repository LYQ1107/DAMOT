"""No stopping gate; exact HF trainer with at most two idle-card frozen jobs."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
Q=Path(__file__).resolve().parents[1]
PY='/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'
DOMAINS=['overcast','rainy','dawn_dusk','night']
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def state(status,**extra):dump(Q/'manifests/controller.json',dict(status=status,pid=os.getpid(),time=time.time(),**extra))
def alive(pid):
    try:return Path('/proc/%d/stat'%pid).read_text().split()[2]!='Z'
    except FileNotFoundError:return False
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def infer(stage,domain):
    name='eval_stage%d_%s'%(stage,domain)
    args=[sys.executable,str(Q/'scripts/gpu_job.py'),'--name',name,'--gpus','1',
        '--prefer','5,7,9,4,2,3,6,8','--',PY,str(Q/'code/tools/run/evaluate_apr_complement.py'),
        '--run-root',str(Q),'--stage',str(stage),'--domain',domain]
    with (Q/'logs'/(name+'.log')).open('w') as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    return stage,domain
def check_cell(stage,domain):
    import csv
    c=Q/'results'/('stage%d'%stage)/domain
    if not (c/'complete.json').exists():return False
    done=json.loads((c/'complete.json').read_text());assert done['status']=='complete' and done['all_frame_HOTA']
    snapshot=json.loads((Q/'episode/state'/('stage%d_complete.json'%stage)).read_text())
    rows=list(csv.DictReader((c/'metrics.csv').open()));assert len(rows)==7
    assert all(r['checkpoint_sha256']==snapshot['checkpoint_sha256'] for r in rows)
    return True
def main():
    while not (Q/'preflight/validation.json').exists():
        p=Q/'preflight/state.json'
        if p.exists() and json.loads(p.read_text())['status']=='failed':
            state('preflight_failed_no_formal_training');return
        state('waiting_for_preflight');time.sleep(5)
    assert json.loads((Q/'preflight/validation.json').read_text())['status']=='passed'
    code=json.loads((Q/'preflight/V3_code_snapshot.json').read_text())
    for item in code['files']:assert sha(Q/'code'/item['path'])==item['sha256']
    assert shutil.disk_usage(Q).free>12*2**30,'Storage headroom must be restored before formal start'
    dispatch=Q/'manifests/formal_dispatch.json';proc=None
    if dispatch.exists() and alive(json.loads(dispatch.read_text())['pid']):
        pid=json.loads(dispatch.read_text())['pid']
    elif (Q/'episode/train_complete.json').exists():pid=-1
    else:
        previous=Q/'manifests/formal_job.json'
        assert not previous.exists() or json.loads(previous.read_text())['status']!='failed','Inspect and record recovery before relaunch'
        with socket.socket() as sock:sock.bind(('',0));port=sock.getsockname()[1]
        args=[sys.executable,str(Q/'scripts/gpu_job.py'),'--name','formal','--gpus','4','--prefer','2,3,6,8,9','--',
            PY,'-m','torch.distributed.launch','--nproc_per_node=4','--master_port='+str(port),'--use_env',str(Q/'scripts/run_noics.py')]
        with (Q/'logs/formal.log').open('w') as log:proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        pid=proc.pid;dump(dispatch,dict(pid=pid,command=args,time=time.time()))
    state('training_and_early_frozen_evaluation',training_job_pid=pid)
    scheduled={};completed=[];verified_stages=set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            for stage in range(1,5):
                p=Q/'episode/state'/('stage%d_complete.json'%stage)
                if not p.exists():continue
                s=json.loads(p.read_text());assert s['epoch']==5
                if stage not in verified_stages:
                    assert sha(Path(s['checkpoint']))==s['checkpoint_sha256'];verified_stages.add(stage)
                for domain in DOMAINS[:stage]:
                    key=(stage,domain)
                    if key not in scheduled:
                        if check_cell(stage,domain):scheduled[key]=None;completed.append(key)
                        else:scheduled[key]=pool.submit(infer,stage,domain)
            for key,future in list(scheduled.items()):
                if future is not None and future.done():
                    future.result();assert check_cell(*key);completed.append(key);scheduled[key]=None
            dump(Q/'manifests/evaluation_queue.json',dict(completed=completed,scheduled=list(scheduled),
                running_or_queued=[k for k,f in scheduled.items() if f is not None],max_concurrent_inference=2,time=time.time()))
            if pid<=0 or not alive(pid):
                if proc is not None:assert proc.wait()==0
                assert json.loads((Q/'manifests/formal_job.json').read_text())['status']=='complete'
                assert (Q/'episode/train_complete.json').exists()
                if len(completed)==10:break
                state('finishing_frozen_evaluation',completed_cells=len(completed))
            time.sleep(5)
    state('video_pair_statistics')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_THREADING_LAYER='GNU')
    for label in ['HF','NoICS']:
        for domain in DOMAINS:
            with (Q/'logs'/('video_%s_%s.log'%(label,domain))).open('w') as log:
                subprocess.run([PY,str(Q/'scripts/tracking_per_video.py'),'--model',label,'--domain',domain,
                    '--source',str((Q if label=='NoICS' else Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420'))/'results/stage4'/domain)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([PY,str(Q/'scripts/video_statistics.py')],env=env,check=True)
    subprocess.run([sys.executable,str(Q/'scripts/report.py')],check=True)
    subprocess.run([sys.executable,str(Q/'scripts/validate_completed.py')],check=True)
    state('complete_awaiting_publication')
if __name__=='__main__':
    try:main()
    except BaseException:
        state('failed',error=traceback.format_exc());raise
