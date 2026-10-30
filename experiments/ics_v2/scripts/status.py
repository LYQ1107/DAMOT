"""Bounded local status reads; never traverse the NAS or load weights."""
from common import *
import shutil

def read(p):
    return json.loads(p.read_text()) if p.exists() else None

status = dict(controller=read(Q / 'manifests/controller.json'),
              historical=read(Q / 'manifests/historical_inference_state.json'),
              historical_cells=sum(1 for p in (Q / 'historical_inference').glob('*/*/*/complete.json')),
              free_GiB=round(shutil.disk_usage(str(Q)).free / 2**30, 2))
status['training'] = []
for group in ['C0','C1','C2','C3','C4']:
    root = Q / 'runs' / group
    p = next((p for p in [root / 'v2_end_rank0.json',root / 'v2_progress_rank0.json',root / 'v2_start_rank0.json'] if p.exists()), None)
    if p:
        value = read(p)
        # CSV gives current iteration between the ten-step progress snapshots.
        rows = list(csv.DictReader((root / 'iterations_rank0.csv').open()))
        status['training'].append(dict(group=group, tag=value['tag'], steps=len(rows),
             key_ratio=value['key_effective_ratio'], ref_ratio=value['ref_effective_ratio'],
             first_init=value['first_initialized_iteration'], updates=value['num_updates']))
status['evaluation_cells'] = sum((Q / 'evaluation' / g / d / 'complete.json').exists()
    for g in ['X3-0','X3-1','C0','C1','C2','C3','C4'] for d in ['night','dawn_dusk'])
status['evaluation_progress'] = []
for group in ['C0','C1','C2','C3','C4']:
    for domain,total in [('night',36),('dawn_dusk',33)]:
        path = Q / 'logs' / (group+'_eval_'+domain+'.log')
        if path.exists():
            status['evaluation_progress'].append(dict(group=group,domain=domain,
                videos_complete=path.read_text().count('VIDEO_COMPLETE'),videos_total=total,
                metrics_complete=(Q / 'evaluation' / group / domain / 'complete.json').exists()))
status['active_jobs'] = [dict(name=p.stem, status=r['status'], gpus=r.get('gpus'))
    for p in sorted((Q / 'manifests').glob('*_job.json')) for r in [read(p)]
    if r.get('status') not in ['complete','failed']]
status['activation'] = []
for group in ['C1','C2','C3','C4']:
    ranks = []
    for rank in range(4):
        p = Q / 'runs' / group / ('iterations_rank%d.csv' % rank)
        if p.exists():
            values = list(csv.DictReader(p.open()))
            if values:
                ranks.append(dict(n=len(values),key=sum(r['key_trim_effective']=='True' for r in values),
                                  ref=sum(r['ref_trim_effective']=='True' for r in values)))
    if len(ranks)==4:
        n = sum(r['n'] for r in ranks)
        item = dict(group=group,min_steps=min(r['n'] for r in ranks),
                    key_ratio=sum(r['key'] for r in ranks)/n,ref_ratio=sum(r['ref'] for r in ranks)/n,
                    min_rank_key_ratio=min(r['key']/r['n'] for r in ranks))
        early = [Q / 'runs' / group / ('v2_iteration50_rank%d.json' % rank) for rank in range(4)]
        if all(p.exists() for p in early):
            data = [read(p) for p in early]
            count = sum(r['trim_active_batches'] for r in data)
            item['first50'] = dict(key_ratio=sum(r['key_effective_batches'] for r in data)/200,
                ref_ratio=sum(r['ref_effective_batches'] for r in data)/200,
                min_rank_key_ratio=min(r['key_effective_ratio'] for r in data),
                trim_loss_mean=sum(r['loss_track_trim_mean']*r['trim_active_batches'] for r in data)/count,
                first_init=[r['first_initialized_iteration'] for r in data])
        status['activation'].append(item)
print(json.dumps(status))
