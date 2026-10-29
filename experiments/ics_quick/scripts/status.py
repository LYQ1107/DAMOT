"""Compact live status, without touching weights or the mounted input tree."""
from common import *
import time
status=json.loads((Q/'manifests/result_status.json').read_text())
active=[]
for p in sorted((Q/'manifests').glob('*_job.json')):
    r=json.loads(p.read_text())
    if r.get('status') not in ('running','waiting_for_idle_gpus'):continue
    proc=Path('/proc')/str(r.get('pid',0))/'stat'
    if not proc.exists() or proc.read_text().split()[2]=='Z':continue
    active.append(dict(job=p.stem[:-4],status=r['status'],gpus=r.get('gpus'),available=r.get('available')))
progress=[];training=[]
for model in ['F','HF']:
    for domain in ['overcast','night']:
        for beta in [0,1]:
            cell=Q/'X2'/model/domain/('beta%d'%beta)
            if not cell.exists():continue
            progress.append(dict(model=model,domain=domain,beta=beta,videos=len(list((cell/'predictions').glob('*.pkl'))),complete=(cell/'complete.json').exists()))
fs=os.statvfs(str(Q));status.update(active_jobs=active,X2=progress,free_disk_GiB=round(fs.f_bavail*fs.f_frsize/2**30,2))
for group in ['X3-0','X3-1','X3-2','X3-3']:
    root=Q/'X3'/group
    for name in ['audit_end_rank0.json','audit_progress_rank0.json','audit_start_rank0.json']:
        p=root/name
        if p.exists():
            r=json.loads(p.read_text());training.append(dict(group=group,tag=r['tag'],epoch=r['epoch'],iteration=r['iteration'],trim_batches=r['trim_triggered_batches'],updates=r['num_updates']));break
status['X3_training']=training
print(json.dumps(status,ensure_ascii=False))
