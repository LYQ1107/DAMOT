"""Stage only this task's results and its two explicitly requested modules."""
import hashlib
import json
import shutil
from pathlib import Path

W = Path(__file__).resolve().parents[1]
root = W / 'publication'
dest = root / 'experiments/idapr'
if dest.exists():
    shutil.rmtree(str(dest))
dest.mkdir(parents=True)
allowed = {'.csv', '.json', '.txt', '.log', '.py', '.npy', '.md', '.patch'}
for top in ['A', 'manifests', 'logs', 'scripts']:
    for p in sorted((W / top).rglob('*')):
        if not p.is_file() or p.is_symlink() or p.suffix not in allowed:
            continue
        if '__pycache__' in p.parts or (top == 'manifests' and p.name.startswith('github_')):
            continue
        target = dest / p.relative_to(W)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(p), str(target))
for name in ['SUMMARY.md', 'README.txt', 'inputs.txt']:
    shutil.copy2(str(W / name), str(dest / name))
for name in ['idapr_basis.py', 'idapr_offline.py']:
    rel = Path('tools/analysis') / name
    for target in [root / rel, dest / 'source' / rel]:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(W / 'code' / rel), str(target))
(dest / '.gitignore').write_text('*.npz\n*.pth\n*.pkl\n__pycache__/\n')
files = []
for p in sorted(root.rglob('*')):
    if not p.is_file():
        continue
    assert p.suffix not in {'.npz', '.pth', '.pkl'}
    data = p.read_bytes()
    files.append(dict(path=str(p.relative_to(root)), source=str(p), bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest(),
                      git_blob_sha=hashlib.sha1(('blob %d\0' % len(data)).encode() + data).hexdigest()))
(W / 'manifests/github_upload_files.json').write_text(json.dumps(files, indent=2) + '\n')
print(json.dumps(dict(files=len(files), unique_blobs=len({r['git_blob_sha'] for r in files}),
                      bytes=sum(r['bytes'] for r in files), forbidden_files=[])))
