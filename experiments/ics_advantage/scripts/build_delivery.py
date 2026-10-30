"""Freeze reproducible evidence; exclude images, model weights and feature tensors."""
import shutil
from common import *

def main():
    dest=ROOT/'delivery/experiments/ics_advantage';dest.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'SUMMARY.md',ROOT/'README.txt']
    for directory in ['analysis','tracking','figures','scripts','manifests','logs']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix not in ['.npz','.pth','.pkl','.pyc'] and '__pycache__' not in str(p)
                     and p!=ROOT/'manifests/delivery_index.json')
    for directory in ['features','tsne_features','vk_features','smoke_features']:
        if (ROOT/directory).exists():files.extend((ROOT/directory).rglob('*meta.json'))
    if (ROOT/'inputs/model_bases').exists():files.extend((ROOT/'inputs/model_bases').glob('*.npy'))
    sources=[]
    for directory in ['code','code_online']:
        if (ROOT/directory).exists():
            for p in (ROOT/directory).rglob('*.py'):
                if '__pycache__' not in str(p):files.append(p);sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    for path in files:
        if not path.exists():continue
        target=dest/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(path),str(target))
    dump(dest/'manifests/model_source_hashes.json',sources)
    index=dest/'manifests/delivery_index.json'
    payload=[dict(path=str(p.relative_to(ROOT/'delivery')),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and p!=index]
    dump(index,payload);shutil.copy2(str(index),str(ROOT/'manifests/delivery_index.json'))
    checks=sorted(payload+[dict(path=str(index.relative_to(ROOT/'delivery')),sha256=sha(index))],key=lambda r:r['path'])
    (dest/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in checks))
    print('FROZEN_DELIVERY_FILES',len(checks)+1,flush=True)

if __name__=='__main__':main()
