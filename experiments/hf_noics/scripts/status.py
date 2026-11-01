import csv,json,shutil
from pathlib import Path
Q=Path(__file__).resolve().parents[1]
def read(path):
    p=Q/path
    return json.loads(p.read_text()) if p.exists() else None
out=dict(root=str(Q),free_GiB=round(shutil.disk_usage(Q).free/2**30,2),controller=read('manifests/controller.json'),
    preflight=read('preflight/state.json'),progress=read('progress.json'),validation=read('preflight/validation.json'))
out['jobs']=[]
for p in sorted((Q/'manifests').glob('*_job.json')):
    v=json.loads(p.read_text());out['jobs'].append({k:v.get(k) for k in ['status','pid','gpus','available','exit_code']});out['jobs'][-1]['name']=p.stem
out['cells']=[]
for stage in range(1,5):
    for domain in ['overcast','rainy','dawn_dusk','night'][:stage]:
        if (Q/'results'/('stage%d'%stage)/domain/'complete.json').exists():out['cells'].append([stage,domain])
out['smoke']={}
for label in ['HF','NoICS']:
    records=[]
    for p in sorted((Q/'preflight/smoke'/label/'audit').glob('stage*_rank0_summary.json')):
        v=json.loads(p.read_text());records.append({k:v[k] for k in ['stage','calls','APR_active_iterations','history_domain_count','bank_none']})
    out['smoke'][label]=records
out['decision']=read('results/decision.json')
print(json.dumps(out))
