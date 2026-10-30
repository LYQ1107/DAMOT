"""Persistent inference queue, limited to explicitly authorized shared GPU0/1."""
import subprocess,time
from common import *
jobs=[(m,d) for d in DOMAINS for m in ['C2','C4','C0']]
active={};done=[];failed=[]
assert json.loads((ROOT/'manifests/selftest.json').read_text())['status']=='PASS'
while jobs or active:
    if not os.path.ismount('/data1/liuyeqiang/nas/A100'):
        time.sleep(15);continue
    for gpu,info in list(active.items()):
        rc=info['process'].poll()
        if rc is None:continue
        info['log'].close();model,domain=info['job'];phase=info['phase'];del active[gpu]
        if rc:
            failed.append(dict(model=model,domain=domain,phase=phase,returncode=rc))
        elif phase=='preflight':
            jobs.insert(0,(model,domain,'full',gpu))
        else:done.append(dict(model=model,domain=domain))
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
    free={int(line.split(',')[0]):int(line.split(',')[1]) for line in status.strip().splitlines()}
    for gpu in [0,1]:
        if gpu in active or free[gpu]<12000 or not jobs:continue
        ix=next((i for i,j in enumerate(jobs) if len(j)==2 or j[3]==gpu),None)
        if ix is None:continue
        job=jobs.pop(ix);model,domain=job[:2];phase='preflight' if len(job)==2 else 'full'
        if (ROOT/'jobs'/model/domain/'inference_complete.json').exists():done.append(dict(model=model,domain=domain));continue
        cmd=[PY,'-u',str(ROOT/'scripts/infer.py'),'--model',model,'--domain',domain]
        if phase=='preflight':cmd.append('--preflight')
        log=(ROOT/'logs'/('%s_%s_%s.log'%(model,domain,phase))).open('ab',buffering=0)
        proc=subprocess.Popen(cmd,env=environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        active[gpu]=dict(process=proc,log=log,job=(model,domain),phase=phase)
        print('START',gpu,model,domain,phase,proc.pid,flush=True)
    dump(ROOT/'manifests/controller_status.json',dict(status='running' if jobs or active else 'inference_done',
         pending=[list(j) for j in jobs],active=[dict(gpu=g,model=v['job'][0],domain=v['job'][1],phase=v['phase'],pid=v['process'].pid) for g,v in active.items()],
         completed=done,failures=failed,updated_unix=time.time()))
    if jobs or active:time.sleep(10)
if failed:raise RuntimeError(failed)
print('BATCH1_INFERENCE_COMPLETE',flush=True)
