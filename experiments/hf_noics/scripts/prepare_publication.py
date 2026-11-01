"""Publish only this experiment's complete reviewable evidence and exact HF source."""
import hashlib
import json
from pathlib import Path
import re
import shutil
Q=Path(__file__).resolve().parents[1]
assert json.loads((Q/'manifests/final_validation.json').read_text())['status']=='passed'
assert json.loads((Q/'results/decision.json').read_text())['status']=='complete'
dest=Q/'publication/experiments/hf_noics'
if dest.exists():shutil.rmtree(str(dest))
dest.mkdir(parents=True)
allowed={'.py','.json','.csv','.txt','.md','.log','.patch','.npy','.pdf','.png'}
compacted=[];json_lines=[];incomplete=[]
progress=re.compile(r'^\[[>\s]*\]\s+\d+/\d+,\s+.*?ETA:\s*(?:\d+s|\?+s?)')
for name in ['configs','manifests','logs','preflight','results','scripts','analysis_tools','episode','audit','tracking','reference','optional_drift']:
    if not (Q/name).exists():continue
    for p in sorted((Q/name).rglob('*')):
        if p.is_symlink() or not p.is_file() or p.suffix not in allowed:continue
        if '__pycache__' in p.parts or 'predictions' in p.parts:continue
        if p.name.startswith('github_') or p.name.endswith('_coco.json'):continue
        target=dest/p.relative_to(Q);target.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix=='.json':
            try:
                raw=p.read_text()
                if p.name.endswith('.log.json'):
                    records=[json.loads(line) for line in raw.splitlines() if line.strip()]
                    assert records and all(isinstance(r,dict) for r in records)
                    json_lines.append(dict(path=str(p.relative_to(Q)),records=len(records),original_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
                else:json.loads(raw)
            except json.JSONDecodeError:
                assert any(part.startswith('failed_') for part in p.parts),str(p)
                target=target.with_name(target.name+'.incomplete.txt')
                incomplete.append(dict(original_path=str(p.relative_to(Q)),published_path=str(target.relative_to(dest)),original_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        if p.suffix=='.log':
            data=p.read_bytes();kept=[];removed=0
            for line in data.decode('utf8').splitlines():
                match=progress.match(line)
                if match:removed+=1;line=line[match.end():]
                if line or not match:kept.append(line)
            if removed:
                target.write_text('\n'.join(kept)+'\n')
                compacted.append(dict(path=str(p.relative_to(Q)),original_sha256=hashlib.sha256(data).hexdigest(),removed_progress_prefixes=removed));continue
        shutil.copy2(str(p),str(target))
for name in ['SUMMARY.md','README.txt','config_diff.json']:shutil.copy2(str(Q/name),str(dest/name))
for p in sorted((Q/'code').rglob('*')):
    if p.is_symlink() or not p.is_file() or '__pycache__' in p.parts:continue
    target=dest/'source'/p.relative_to(Q/'code');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(p),str(target))
(dest/'manifests/publication_log_formats.json').write_text(json.dumps(dict(original_logs_retained=True,compacted_progress_prefixes=compacted,MMCV_JSON_lines_logs=json_lines,incomplete_failed_attempt_archives=incomplete),indent=2)+'\n')
files=[]
for p in sorted(dest.rglob('*')):
    if not p.is_file():continue
    data=p.read_bytes();files.append(dict(path=str(p.relative_to(Q/'publication')),source=str(p),bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),git_blob_sha=hashlib.sha1(('blob %d\0'%len(data)).encode()+data).hexdigest()))
(Q/'manifests/github_upload_files.json').write_text(json.dumps(files,indent=2)+'\n')
print('STAGED',len(files),sum(f['bytes'] for f in files))
