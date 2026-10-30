"""Copy only task evidence (not data/weights) for the staged GitHub publication."""
import shutil
from pathlib import Path
from evidence_cpu import ROOT, dump, sha


def main():
    dest=ROOT/'delivery/experiments/ics_evidence'
    dest.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'SUMMARY.md',ROOT/'vkitti2_inputs.txt']
    for directory in ['part1','part2','part3','scripts','manifests','logs']:
        if (ROOT/directory).exists():
            paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and
                         p.suffix not in ['.npz','.pth','.pyc'] and '__pycache__' not in str(p))
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
    records=[dict(path=str(p.relative_to(ROOT/'delivery')),bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (dest/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in records))
    dump(ROOT/'manifests/delivery_index.json',records)
    print('Delivery files',len(records)+1,flush=True)


if __name__=='__main__':
    main()
