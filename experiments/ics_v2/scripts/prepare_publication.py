"""Stage completed outputs under experiments/ics_v2 only, without weights or raw data."""
from common import *
import shutil
import re

assert json.loads((Q / 'manifests/final_validation.json').read_text())['status']=='passed'
assert json.loads((Q / 'manifests/result_status.json').read_text())['complete']
assert len(json.loads((Q / 'manifests/final_model_provenance.json').read_text())['models']) == 5
destination = Q / 'publication/experiments/ics_v2'
if destination.exists():
    shutil.rmtree(str(destination))
destination.mkdir(parents=True)
allowed = {'.csv','.json','.py','.patch','.txt','.log','.md','.npy','.yml','.yaml'}
progress_pattern = re.compile(r'^\[[>\s]*\]\s+\d+/\d+,\s+.*?ETA:\s*(?:\d+s|\?+s?)')
compacted_logs = []
for top in ['basis','configs','evaluation','runs','manifests','logs','scripts',
            'historical','historical_completed','historical_inference','preflight','references']:
    parent = Q / top
    if not parent.exists():
        continue
    for source in sorted(parent.rglob('*')):
        if not source.is_file() or source.is_symlink() or source.suffix not in allowed:
            continue
        if '__pycache__' in source.parts or 'predictions' in source.parts:
            continue
        if 'superseded' in str(source.relative_to(Q)):
            continue
        if top=='manifests' and source.name.startswith('github_'):
            continue
        if source.name.endswith(('_coco.json','_detailed.csv','_summary.txt')):
            continue
        target = destination / source.relative_to(Q)
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix == '.log':
            original = source.read_bytes()
            kept, removed = [], 0
            for line in original.decode('utf8').splitlines():
                match = progress_pattern.match(line)
                if match:
                    removed += 1
                    line = line[match.end():]
                if line or not match:
                    kept.append(line)
            if removed:
                target.write_text('\n'.join(kept) + '\n')
                compacted_logs.append(dict(path=str(source.relative_to(Q)),
                     original_bytes=len(original),original_sha256=sha(source),
                     published_bytes=target.stat().st_size,published_sha256=sha(target),
                     removed_frame_progress_records=removed))
            else:
                shutil.copy2(str(source),str(target))
        else:
            shutil.copy2(str(source),str(target))
dump(destination / 'manifests/publication_log_compaction.json',dict(
     method='Only per-frame carriage-return progress prefixes removed; other messages retained.',
     original_logs_retained_on_server=True,files=compacted_logs))
for name in ['SUMMARY.md','README.txt','inputs.txt','metrics.csv','criterion.csv',
             'training_audit.csv','activation_first50.csv','code_changes.patch']:
    if (Q / name).exists():
        shutil.copy2(str(Q / name),str(destination / name))
for row in json.loads((Q / 'manifests/code_seal.json').read_text())['files']:
    rel = row['relative_path']
    current, baseline = Q / 'code_hf' / rel, Q / 'code_original' / rel
    assert sha(current)==row['sha256']
    target = destination / 'source' / rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(str(current),str(target))
    if baseline.exists() and current.read_bytes()!=baseline.read_bytes():
        target = destination / 'source_original' / rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(str(baseline),str(target))
(destination / '.gitignore').write_text('*.pth\n*.npz\n*.pkl\n__pycache__/\n')
files = []
for path in sorted(destination.rglob('*')):
    if not path.is_file():
        continue
    assert path.suffix not in ['.pth','.npz','.pkl']
    payload = path.read_bytes()
    files.append(dict(path=str(path.relative_to(Q / 'publication')),source=str(path),bytes=len(payload),
         sha256=hashlib.sha256(payload).hexdigest(),
         git_blob_sha=hashlib.sha1(('blob %d\0'%len(payload)).encode()+payload).hexdigest()))
dump(Q / 'manifests/github_upload_files.json',files)
dump(Q / 'manifests/github_publication_staging.json',dict(files=len(files),bytes=sum(r['bytes'] for r in files),
     prefix='experiments/ics_v2/',weights_or_raw_data=False))
print('PUBLICATION_STAGED',len(files),sum(r['bytes'] for r in files),flush=True)
