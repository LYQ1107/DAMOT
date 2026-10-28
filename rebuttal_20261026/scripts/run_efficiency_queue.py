#!/usr/bin/env python3
import os,time,json,subprocess
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
def wait_for(p):
    while not p.exists():time.sleep(20)
def job(mode,extra,complete):
    if complete.exists():return
    label='efficiency_'+mode+('_'+extra[-1] if extra else '')
    path=RUN/'logs'/(label+'.log')
    if path.exists():path=path.with_name(path.stem+'_'+str(int(time.time()))+'.log')
    cmd=[str(RUN/'env/bin/python'),'-u',str(RUN/'scripts/profile_damot.py'),mode]+extra
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='9',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with open(path,'w') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
        rec=dict(pid=p.pid,gpu=9,command=cmd,start=time.time(),log=str(path))
        (RUN/'manifests'/('process_'+label+'.json')).write_text(json.dumps(rec,indent=2));print('START',label,p.pid,flush=True)
        code=p.wait();rec.update(exit_code=code,end=time.time(),complete=complete.exists())
        (RUN/'manifests'/('process_'+label+'.json')).write_text(json.dumps(rec,indent=2));print('EXIT',label,code,flush=True)
    if code or not complete.exists():raise RuntimeError('job failed '+label)
if __name__=='__main__':
    wait_for(RUN/'results/exp4/reconstructed_history.pkl');wait_for(RUN/'manifests/night_decode_validation.json')
    # Keep the host's own evaluation workers idle during formal performance
    # measurement as well as keeping device9 free of other task jobs.
    wait_for(RUN/'manifests/accuracy_queue_exit.json')
    wait_for(RUN/'manifests/all_frame_hota_complete.json')
    job('throughput',[],RUN/'results/exp4/adaptation_complete.json')
    for n in range(4):job('modules',['--history',str(n)],RUN/'results/exp4'/('module_time_history%d.csv'%n))
    job('inference',[],RUN/'results/exp4/inference_complete.json')
    (RUN/'manifests/efficiency_queue_exit.json').write_text(json.dumps(dict(exit_code=0,completed=True),indent=2))
