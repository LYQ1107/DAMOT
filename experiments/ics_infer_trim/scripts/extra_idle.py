"""Use newly idle7/9 only; per-job locks deduplicate the primary queue."""
from concurrent.futures import ThreadPoolExecutor
import subprocess,time
from common import *
def work(gpu,model):
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True)
        values={int(a):(int(b),int(c)) for a,b,c in [r.split(',') for r in raw.strip().splitlines()]}
        if values[gpu][0]<256 and values[gpu][1]<=5:break
        time.sleep(15)
    for phase in ['preflight','full']:
        if (ROOT/'jobs'/model/'dawn_dusk/inference_complete.json').exists():return
        args=[PY,'-u',str(ROOT/'scripts/infer.py'),'--model',model,'--domain','dawn_dusk']
        if phase=='preflight':args.append('--preflight')
        log=ROOT/'logs'/('%s_dawn_dusk_%s_gpu%d.log'%(model,phase,gpu))
        with log.open('ab',buffering=0) as f:
            subprocess.run(args,env=environment(gpu),stdin=subprocess.DEVNULL,stdout=f,stderr=f,check=True)
    print('EXTRA_IDLE_COMPLETE',gpu,model,flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
    fs=[pool.submit(work,7,'C4'),pool.submit(work,9,'C0')]
    for f in fs:f.result()
