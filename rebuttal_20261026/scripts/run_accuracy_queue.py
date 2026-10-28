#!/usr/bin/env python3
"""Five independent GPU workers; per-GPU jobs are strictly serial and resumable."""
import os,sys,json,time,subprocess,concurrent.futures
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
PY=RUN/'env/bin/python'
def wait_inputs(paths):
    while not all(p.exists() for p in paths):time.sleep(15)
def run_job(gpu,script,args,label,required,complete):
    if complete.exists():return dict(job=label,status='already_complete')
    wait_inputs(required)
    desired=[str(PY),'-u',str(RUN/'scripts'/script)]+args
    # A restarted controller waits on an already running matching job, including
    # a job deliberately migrated to another GPU, instead of launching a copy.
    for record_path in (RUN/'manifests').glob('process_*.json'):
        try:
            old=json.loads(record_path.read_text())
            if old.get('command')!=desired:continue
            pid=old['pid'];proc=Path('/proc')/str(pid)/'cmdline'
            while proc.exists() and proc.read_bytes().split(b'\0')[:-1]==[v.encode() for v in desired]:
                if complete.exists():return dict(job=label,status='existing_job_completed',pid=pid)
                time.sleep(20)
        except (FileNotFoundError,KeyError,ProcessLookupError):pass
    if complete.exists():return dict(job=label,status='already_complete_after_wait')
    log=RUN/'logs'/(label+'.log')
    if log.exists():log=log.with_name(log.stem+'_'+str(int(time.time()))+log.suffix)
    cmd=desired
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with open(log,'w') as f:
        child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=str(RUN/'code'))
        rec=dict(job=label,pid=child.pid,gpu=gpu,command=cmd,log=str(log),start=time.time())
        (RUN/'manifests'/('process_'+label+'.json')).write_text(json.dumps(rec,indent=2))
        print('START',label,'gpu',gpu,'pid',child.pid,flush=True)
        code=child.wait();rec.update(exit_code=code,end=time.time(),complete=complete.exists())
        (RUN/'manifests'/('process_'+label+'.json')).write_text(json.dumps(rec,indent=2))
        print('EXIT',label,code,'complete',complete.exists(),flush=True)
    return rec
def worker(gpu,model,tp):
    recs=[run_job(gpu,'evaluate_clear.py',['--model',model],'clear_'+model,[RUN/'cache/weights'/(model+'.pth'),RUN/'manifests/clear_holdout_45v_decode_validation.json'],RUN/'results/exp2'/model/'complete.json')]
    if tp:
        ck={'e0':'F_stage3_e5','e1':'F_stage4_e1','e3':'F_stage4_e3','e5':'F_stage4_e5'}[tp]
        recs.append(run_job(gpu,'diagnose_teacher.py',['--timepoint',tp],'teacher_'+tp,[RUN/'cache/weights'/(ck+'.pth'),RUN/'manifests/night_decode_validation.json'],RUN/'results/exp3'/tp/'complete.json'))
    return recs
if __name__=='__main__':
    print('QUEUE PID',os.getpid(),flush=True)
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        fs=[pool.submit(worker,gpu,m,tp) for gpu,m,tp in [(4,'S','e0'),(5,'F_stage1_e5','e1'),(6,'F_stage2_e5','e3'),(7,'F_stage3_e5','e5'),(8,'F_stage4_e5',None)]]
        records=[]
        for f in concurrent.futures.as_completed(fs):
            try:records.extend(f.result())
            except Exception as e:records.append(dict(exception=repr(e)))
    (RUN/'manifests/accuracy_queue_exit.json').write_text(json.dumps(records,indent=2));print('QUEUE FINISHED',flush=True)
