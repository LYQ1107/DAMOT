"""Compact status; never traverse the private NAS or load checkpoint tensors."""
from common import *
result = json.loads((Q / 'manifests/result_status.json').read_text()) if (Q / 'manifests/result_status.json').exists() else {}
result['controller'] = json.loads((Q / 'manifests/controller.json').read_text())
result['completed_evaluation_cells'] = sum(
    (Q / 'evaluation' / group / domain / 'complete.json').exists()
    for group in ['B0', 'R', 'B3', 'HF_stage3']
    for domain in (['overcast', 'rainy', 'dawn_dusk'] if group == 'HF_stage3'
                   else ['night', 'overcast', 'rainy', 'dawn_dusk']))
result['training'] = []
for group in ['B3', 'R']:
    root = Q / 'runs' / group
    for name in ['apr_audit_end_rank0.json', 'apr_audit_progress_rank0.json', 'apr_audit_start_rank0.json']:
        path = root / name
        if path.exists():
            data = json.loads(path.read_text())
            result['training'].append(dict(group=group, tag=data['tag'], batches=data['total_batches'],
                                           epoch=data['epoch'], mode=data['mode'], coupling=data['coupling'],
                                           updates=data['num_updates']))
            break
result['active_jobs'] = []
for path in sorted((Q / 'manifests').glob('*_job.json')):
    row = json.loads(path.read_text())
    if row['status'] in ['running', 'waiting_for_idle_gpus']:
        proc = Path('/proc') / str(row.get('pid', 0)) / 'stat'
        if proc.exists() and proc.read_text().split()[2] != 'Z':
            result['active_jobs'].append(dict(name=path.stem, status=row['status'],
                                             gpus=row.get('gpus'), available=row.get('available')))
fs = os.statvfs(str(Q))
result['free_GiB'] = round(fs.f_bavail * fs.f_frsize / 2**30, 2)
print(json.dumps(result))
