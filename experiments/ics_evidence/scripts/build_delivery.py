"""Copy only task evidence (not data/weights) for the staged GitHub publication."""
import shutil
from pathlib import Path
from evidence_cpu import ROOT, dump, sha


def main():
    dest=ROOT/'delivery/experiments/ics_evidence'
    dest.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'SUMMARY.md',ROOT/'README.txt',ROOT/'vkitti2_inputs.txt']
    for directory in ['part1','part2','part3','scripts','manifests','logs']:
        if (ROOT/directory).exists():
            paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and
                         p.suffix not in ['.npz','.pth','.pyc'] and '__pycache__' not in str(p)
                         and p != ROOT/'manifests/delivery_index.json')
    if (ROOT/'features').exists():
        paths.extend((ROOT/'features').rglob('*meta.json'))
    paths.append(ROOT/'code/tools/analysis/ics_probe_extract.py')
    for p in paths:
        relative=p.relative_to(ROOT)
        target=dest/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(str(p),str(target))
    # Source snapshot fingerprints make the external model implementations explicit.
    source=[dict(path=str(p.relative_to(ROOT/'code')),sha256=sha(p))
            for p in sorted((ROOT/'code').rglob('*.py')) if '__pycache__' not in str(p)]
    dump(dest/'manifests/model_source_hashes.json',source)
    # The index describes this snapshot's payload, excluding itself and SHA256SUMS.
    # Generate it after copying, so a previous build's index cannot be published.
    index=dest/'manifests/delivery_index.json'
    records=[dict(path=str(p.relative_to(ROOT/'delivery')),bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and p!=index]
    dump(index,records)
    shutil.copy2(str(index),str(ROOT/'manifests/delivery_index.json'))
    checks=sorted(records+[dict(path=str(index.relative_to(ROOT/'delivery')),sha256=sha(index))],key=lambda r:r['path'])
    (dest/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in checks))
    print('Delivery files',len(checks)+1,flush=True)


if __name__=='__main__':
    main()
