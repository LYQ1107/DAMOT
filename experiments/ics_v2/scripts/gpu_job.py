"""Start a local job only when the requested number of GPUs is wholly idle."""
import argparse,subprocess,os,time,fcntl,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpus',type=int,default=1);p.add_argument('--name',required=True);p.add_argument('--prefer',default='2,8,9,3,4,5,6');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
q=Path(__file__).resolve().parents[1];(q/'locks').mkdir(exist_ok=True)
def available():
    txt=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True)
    free={int(r.split(',')[0]) for r in txt.splitlines() if all(int(v.strip())<10 for v in r.split(',')[1:])}
    return [int(v) for v in a.prefer.split(',') if int(v) in free],txt
def record(obj):
    tmp=q/'manifests'/(a.name+'_job.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(q/'manifests'/(a.name+'_job.json'))
while True:
    free,snapshot=available();locks=[];chosen=[]
    for idx in free:
        f=(q/'locks'/('gpu%d.lock'%idx)).open('a')
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:f.close();continue
        locks.append(f);chosen.append(idx)
        if len(chosen)==a.gpus:break
    if len(chosen)==a.gpus:
        check,snapshot=available()
        if all(idx in check for idx in chosen):break
    for f in locks:f.close()
    record(dict(status='waiting_for_idle_gpus',required=a.gpus,available=free,time=time.time(),pid=os.getpid()))
    time.sleep(30)
cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
started=dict(status='running',gpus=chosen,prelaunch_snapshot=snapshot,command=cmd,pid=os.getpid(),time=time.time())
record(started)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,chosen)),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONPATH=str(q/'code_hf')+os.pathsep+os.environ.get('PYTHONPATH',''))
proc=subprocess.Popen(cmd,cwd=str(q/'code_hf'),env=env);code=proc.wait()
record(dict(started,status='complete' if code==0 else 'failed',exit_code=code,finished_time=time.time()));sys.exit(code)
