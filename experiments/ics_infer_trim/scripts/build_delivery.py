"""Freeze source, all CSVs, logs and metadata; omit checkpoints and raw predictions."""
import shutil
from common import *
assert json.loads((ROOT/'manifests/validation_batch1.json').read_text())['status']=='PASS'
dest=ROOT/'delivery/experiments/ics_infer_trim'
if dest.exists():shutil.rmtree(str(dest))
dest.mkdir(parents=True)
sources=[ROOT/'SUMMARY.md']
for folder in ['scripts','code','inputs','manifests','logs','analysis','jobs','results']:
    base=ROOT/folder
    for p in (Path(parent)/name for parent,dirs,files in os.walk(str(base),followlinks=False) for name in files):
        if p.is_symlink() or not p.is_file() or '__pycache__' in p.parts:continue
        if folder=='code' and p.suffix not in ['.py','.md','.txt','.sh','.yml','.yaml','.json']:continue
        if folder=='results' and p.suffix not in ['.csv','.py','.json']:continue
        if folder=='results' and p.suffix=='.json' and p.name not in ['evaluation_flat_metrics.json','metrics_complete.json']:continue
        if folder=='inputs' and p.suffix not in ['.npy','.json']:continue
        if folder=='scripts' and p.suffix!='.py':continue
        sources.append(p)
index=[]
for p in sorted(set(sources)):
    rel=p.relative_to(ROOT);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(str(p),str(q))
    index.append(dict(path=str(rel),bytes=q.stat().st_size,sha256=sha(q)))
dump(dest/'FILE_INDEX.json',dict(files=index,excluded=['model checkpoints','images','raw per-frame predictions and large evaluation JSON'],raw_results_retained=str(ROOT/'results')))
checks=[str(p) for p in sorted(dest.rglob('*')) if p.is_file()]
(dest/'SHA256SUMS.txt').write_text(''.join(sha(Path(p))+'  '+str(Path(p).relative_to(dest))+'\n' for p in checks))
print('DELIVERY_FROZEN',len(checks)+1,str(dest),flush=True)
