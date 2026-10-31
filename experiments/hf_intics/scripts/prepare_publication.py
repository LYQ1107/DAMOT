"""Stage only this task's reviewable files under experiments/hf_intics/."""
import hashlib
import json
from pathlib import Path
import re
import shutil
Q=Path(__file__).resolve().parents[1]
validation=json.loads((Q/'manifests/final_validation.json').read_text())
assert validation['status']=='passed'
clarification=Q/'manifests/angle_clarification.json'
assert not clarification.exists() or json.loads(clarification.read_text())['status']!='pending', 'Angle-definition clarification must be resolved before final publication'
dest=Q/'publication/experiments/hf_intics'
if dest.exists():shutil.rmtree(str(dest))
dest.mkdir(parents=True)
allowed={'.py','.json','.csv','.txt','.md','.log','.patch','.npy','.pdf','.png'}
progress=re.compile(r'^\[[>\s]*\]\s+\d+/\d+,\s+.*?ETA:\s*(?:\d+s|\?+s?)')
compacted=[]
incomplete=[]
json_lines=[]
for name in ['configs','manifests','logs','preflight','probe','probe_tools','results','scripts','episode']:
    for p in sorted((Q/name).rglob('*')):
        if p.is_symlink() or not p.is_file() or p.suffix not in allowed:continue
        if '__pycache__' in p.parts or 'predictions' in p.parts or 'superseded' in str(p):continue
        if p.name.startswith('github_') or p.name.endswith('_coco.json') or p.name=='online_residual.npy':continue
        target=dest/p.relative_to(Q);target.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix=='.json':
            try:
                raw=p.read_text()
                if p.name.endswith('.log.json'):
                    records=[json.loads(line) for line in raw.splitlines() if line.strip()]
                    assert records and all(isinstance(record,dict) for record in records),str(p)
                    json_lines.append(dict(path=str(p.relative_to(Q)),format='MMCV JSON lines',
                        records=len(records),original_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
                else:json.loads(raw)
            except json.JSONDecodeError:
                assert any(part.startswith('failed_') for part in p.parts),str(p)
                target=target.with_name(target.name+'.incomplete.txt')
                incomplete.append(dict(original_path=str(p.relative_to(Q)),
                    published_path=str(target.relative_to(dest)),bytes=p.stat().st_size,
                    original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    reason='Raw metadata truncated by the documented failed attempt; original retained'))
        if p.suffix=='.log':
            data=p.read_bytes();kept=[];removed=0
            for line in data.decode('utf8').splitlines():
                match=progress.match(line)
                if match:removed+=1;line=line[match.end():]
                if line or not match:kept.append(line)
            if removed:
                target.write_text('\n'.join(kept)+'\n')
                compacted.append(dict(path=str(p.relative_to(Q)),original_sha256=hashlib.sha256(data).hexdigest(),removed_progress_records=removed))
                continue
        shutil.copy2(str(p),str(target))
for name in ['SUMMARY.md','README.txt','config_diff.json','code_changes.patch']:
    shutil.copy2(str(Q/name),str(dest/name))
for p in sorted((Q/'code').rglob('*.py')):
    if '__pycache__' in p.parts:continue
    rel=p.relative_to(Q/'code');old=Q/'code_original'/rel
    if old.exists() and old.read_bytes()==p.read_bytes():continue
    target=dest/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(p),str(target))
    if old.exists():
        target=dest/'source_original'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(old),str(target))
(dest/'manifests/publication_log_compaction.json').write_text(json.dumps(dict(original_logs_retained=True,files=compacted),indent=2)+'\n')
(dest/'manifests/publication_incomplete_archives.json').write_text(json.dumps(dict(
    authoritative_results_are_complete=True,original_raw_failed_attempts_retained=True,
    files=incomplete),indent=2)+'\n')
(dest/'manifests/publication_json_lines_logs.json').write_text(json.dumps(dict(
    original_names_and_bytes_preserved=True,files=json_lines),indent=2)+'\n')
files=[]
for p in sorted(dest.rglob('*')):
    if not p.is_file():continue
    data=p.read_bytes()
    files.append(dict(path=str(p.relative_to(Q/'publication')),source=str(p),bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),git_blob_sha=hashlib.sha1(('blob %d\0'%len(data)).encode()+data).hexdigest()))
(Q/'manifests/github_upload_files.json').write_text(json.dumps(files,indent=2)+'\n')
print('STAGED',len(files),sum(r['bytes'] for r in files))
