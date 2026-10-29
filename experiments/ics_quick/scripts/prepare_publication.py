"""Stage a small, immutable publication bundle; never traverse the NAS mount."""
from common import *
import shutil
dest=Q/'publication/experiments/ics_quick'
if dest.exists():shutil.rmtree(str(dest))
dest.mkdir(parents=True)
allowed={'.csv','.json','.py','.patch','.txt','.log','.md','.npy','.yml','.yaml'}
copied=[]
for top in ['X1','X2','X3','configs','logs','manifests','scripts','feats']:
    for source in sorted((Q/top).rglob('*')):
        if not source.is_file() or source.is_symlink() or source.suffix not in allowed:continue
        if '__pycache__' in source.parts:continue
        if top=='manifests' and (source.name.startswith('github_') or source.name=='publication_staging.json'):continue
        # Detailed tracker curves and per-frame predictions remain local;
        # the report includes all aggregate CSVs, logs, bases and configs.
        if any(p in source.parts for p in ['predictions','evaluation/track']):continue
        if source.name.endswith('_coco.json'):continue
        if source.name.endswith('_detailed.csv') or source.name.endswith('_summary.txt'):continue
        target=dest/source.relative_to(Q);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(source),str(target));copied.append(target)
for name in ['SUMMARY.md','inputs.txt','README.txt']:
    shutil.copy2(str(Q/name),str(dest/name));copied.append(dest/name)
(dest/'.gitignore').write_text('*.pth\n*.npz\n*.pkl\n__pycache__/\n')
manifest=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=sha(p)) for p in copied]
dump(dest/'publication_files.json',manifest)
dump(Q/'manifests/publication_staging.json',dict(directory=str(dest),files=len(manifest),bytes=sum(x['bytes'] for x in manifest),forbidden_files=[]))
assert not any(dest.rglob('*.pth')) and not any(dest.rglob('*.npz'))
print('PUBLICATION_STAGED',len(manifest),sum(x['bytes'] for x in manifest),flush=True)
