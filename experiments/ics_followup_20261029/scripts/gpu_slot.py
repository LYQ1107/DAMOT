"""Reserve genuinely idle GPUs and cooperate with existing DAMOT job locks."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=1);ap.add_argument('--prefer',default='8,9,7,2,3,4,5,6');ap.add_argument('--name',required=True);ap.add_argument('--after',type=Path);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
from protocol import write_json
def idle():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True)
    free={int(p[0]) for p in [r.split(',') for r in raw.splitlines()] if int(p[1])<10 and int(p[2])<10}
    if not Path('/data1/liuyeqiang/damot_apr_complement_hf_nogeom_20261029_1443/controller_complete.json').exists():free-={3,4,5,6}
    return free,raw
roots=[ROOT/'locks'];legacy=Path('/data1/liuyeqiang/ics_quick_20261029_160807/locks')
if legacy.exists():roots.append(legacy)
for r in roots:r.mkdir(exist_ok=True)
while True:
    if a.after and not a.after.exists():
        write_json(ROOT/'manifests'/(a.name+'_gpu.json'),dict(status='waiting_for_NOGEOM_completion',marker=str(a.after),pid=os.getpid()));time.sleep(30);continue
    free,raw=idle();locks=[];chosen=[]
    for gpu in [int(v) for v in a.prefer.split(',') if int(v) in free]:
        candidate=[]
        for r in roots:
            f=(r/('gpu%d.lock'%gpu)).open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);candidate.append(f)
            except BlockingIOError:f.close();break
        if len(candidate)==len(roots):locks.extend(candidate);chosen.append(gpu)
        else:
            for f in candidate:f.close()
        if len(chosen)==a.count:break
    check,raw=idle()
    if len(chosen)==a.count and all(v in check for v in chosen):break
    for f in locks:f.close()
    write_json(ROOT/'manifests'/(a.name+'_gpu.json'),dict(status='waiting_for_idle_gpus',required=a.count,available=sorted(free),pid=os.getpid()));time.sleep(30)
cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
cmd=[v.replace('{GPU}',str(chosen[0])) for v in cmd]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,chosen)),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
write_json(ROOT/'manifests'/(a.name+'_gpu.json'),dict(status='running',gpus=chosen,snapshot=raw,command=cmd,pid=os.getpid(),time=time.time()))
p=subprocess.Popen(cmd,env=env,cwd=str(ROOT/'code'));code=p.wait()
write_json(ROOT/'manifests'/(a.name+'_gpu.json'),dict(status='complete' if code==0 else 'failed',exit_code=code,gpus=chosen,command=cmd,pid=os.getpid(),time=time.time()));sys.exit(code)
