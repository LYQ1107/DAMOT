"""Stage known outputs only; never traverse Q/nas or upload weights/NPZs."""
from common import *
import difflib
import shutil

dest = Q / 'publication/experiments/idapr/B3stage'
if dest.exists():
    shutil.rmtree(str(dest))
dest.mkdir(parents=True)
allowed = {'.csv', '.json', '.py', '.patch', '.txt', '.log', '.md', '.npy', '.yml', '.yaml'}
for top in ['basis', 'configs', 'evaluation', 'U', 'runs', 'manifests', 'logs', 'scripts', 'feats', 'source_training']:
    for source in sorted((Q / top).rglob('*')):
        if not source.is_file() or source.is_symlink() or source.suffix not in allowed:
            continue
        if '__pycache__' in source.parts or 'predictions' in source.parts:
            continue
        if top == 'manifests' and source.name.startswith('github_'):
            continue
        if source.name == 'dawn_dusk_images_only.json':
            continue
        if source.name.endswith(('_coco.json', '_detailed.csv', '_summary.txt')):
            continue
        target = dest / source.relative_to(Q)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
for name in ['SUMMARY.md', 'README.txt', 'inputs.txt', 'metrics.csv',
             'mean_forgetting.csv', 'comparison.csv', 'training_audit.csv']:
    if (Q / name).exists():
        shutil.copy2(str(Q / name), str(dest / name))
source_files = [r['relative_path'] for r in json.loads((Q / 'manifests/code_seal.json').read_text())['files']]
source_files.append('tools/analysis/ics_probe_analyze.py')
patches = []
training_patches = []
for rel in source_files:
    current = Q / 'code_hf' / rel
    original = Q / 'code_original' / rel
    target = dest / 'source' / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(current), str(target))
    before = original.read_text().splitlines(True) if original.exists() else []
    after = current.read_text().splitlines(True)
    diff = ''.join(difflib.unified_diff(before, after,
                   fromfile='a/' + rel if original.exists() else '/dev/null', tofile='b/' + rel))
    if diff:
        patches.append(diff)
        if original.exists():
            target = dest / 'source_original' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(original), str(target))
    training_file = Q / 'source_training' / rel
    training_after = training_file.read_text().splitlines(True) if training_file.exists() else after
    training_patches.append(''.join(difflib.unified_diff(before, training_after,
                   fromfile='a/' + rel if original.exists() else '/dev/null', tofile='b/' + rel)))
(dest / 'code_changes.patch').write_text(''.join(patches))
(dest / 'code_training.patch').write_text(''.join(training_patches))
(dest / '.gitignore').write_text('*.pth\n*.npz\n*.pkl\n__pycache__/\n')
files = []
for path in sorted(dest.rglob('*')):
    if not path.is_file():
        continue
    assert path.suffix not in ['.pth', '.npz', '.pkl']
    content = path.read_bytes()
    files.append(dict(path=str(path.relative_to(Q / 'publication')), source=str(path),
                      bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                      git_blob_sha=hashlib.sha1(('blob %d\0' % len(content)).encode() + content).hexdigest()))
dump(Q / 'manifests/github_upload_files.json', files)
dump(Q / 'manifests/github_publication_staging.json', dict(files=len(files), bytes=sum(r['bytes'] for r in files),
     prefix='experiments/idapr/B3stage/', forbidden_files=[]))
print('PUBLICATION_STAGED', len(files), sum(r['bytes'] for r in files), flush=True)
