"""Two CPU workers evaluate completed inference jobs; resume safe."""
import subprocess,time
from common import *
jobs=[(m,c['label'],d) for d in DOMAINS for m in ['C0','C2','C4'] for c in cases(m)]
active={};failed=[]
while jobs or active:
    if not os.path.ismount('/data1/liuyeqiang/nas/A100'):time.sleep(15);continue
    for key,(proc,log) in list(active.items()):
        rc=proc.poll()
        if rc is None:continue
        log.close();del active[key]
        if rc:failed.append(dict(job=list(key),returncode=rc))
    for m,label,d in list(jobs):
        if (ROOT/'results'/label/d/'metrics_complete.json').exists():jobs.remove((m,label,d));continue
        if len(active)>=2:break
        if not (ROOT/'jobs'/m/d/'inference_complete.json').exists():continue
        cmd=[PY,'-u',str(ROOT/'scripts/metrics.py'),'--label',label,'--domain',d]
        log=(ROOT/'logs'/('%s_%s_metrics.log'%(label,d))).open('ab',buffering=0)
        proc=subprocess.Popen(cmd,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        active[m,label,d]=(proc,log);jobs.remove((m,label,d));print('CPU_START',label,d,proc.pid,flush=True)
    dump(ROOT/'manifests/cpu_status.json',dict(pending=[list(j) for j in jobs],active=[dict(job=list(k),pid=v[0].pid) for k,v in active.items()],failures=failed))
    status=ROOT/'manifests/controller_status.json'
    if jobs and not active and status.exists():
        try:s=read_json(status)
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
        if s['status']=='inference_done' and s['failures']:raise RuntimeError('Inference failed; metrics cannot proceed')
    if jobs or active:time.sleep(10)
if failed:raise RuntimeError(failed)
subprocess.run([PY,str(ROOT/'scripts/analysis.py'),'--batch','1'],env=environment(),check=True)
print('BATCH1_ANALYSIS_COMPLETE',flush=True)
