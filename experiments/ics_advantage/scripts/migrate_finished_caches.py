"""Move only this thread's frozen caches to NAS, with SHA256 before removal."""
import hashlib,json,os,shutil,time
from pathlib import Path
BASE=Path('/data1/liuyeqiang');MOUNT=BASE/'nas/A100';STORE=MOUNT/'liuyeqiang_codex_runs'
SOURCES=['ics_evidence_20261030_215311/weights','ics_evidence_20261030_215311/features',
         'ics_evidence_20261030_215311/data/vkitti2/images','ics_advantage_20261031_004629/tsne_features']
def mounted():
    if not os.path.ismount(str(MOUNT)):raise RuntimeError('NAS disconnected; refusing local fallback')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
mounted();journal=STORE/'cache_migration_20261031.json'
records=json.loads(journal.read_text()) if journal.exists() else []
for rel in SOURCES:
    src=BASE/rel;dst=STORE/rel
    if src.is_symlink():
        assert src.resolve()==dst.resolve();continue
    if not src.exists():continue
    frozen=json.loads((BASE/'ics_advantage_20261031_004629/manifests/batch2_status.json').read_text())
    assert frozen['status'].startswith('waiting_'),frozen
    files=[p for p in src.rglob('*') if p.is_file()];total=0
    for p in files:
        mounted();q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True)
        original=sha(p)
        if not q.exists() or q.stat().st_size!=p.stat().st_size or sha(q)!=original:
            temp=q.with_name(q.name+'.copying');shutil.copyfile(str(p),str(temp));mounted()
            assert sha(temp)==original;(temp).replace(q)
        assert sha(q)==original;total+=p.stat().st_size
    mounted();assert len([p for p in dst.rglob('*') if p.is_file()])==len(files)
    link=src.with_name(src.name+'.nas_verified_link');backup=src.with_name(src.name+'.local_verified_backup')
    assert not backup.exists();link.symlink_to(dst,target_is_directory=True)
    src.rename(backup);os.replace(str(link),str(src));mounted()
    shutil.rmtree(str(backup))
    records.append(dict(original=str(src),NAS=str(dst),files=len(files),bytes=total,
                        sha256_verified_every_file=True,path_preserved_by_symlink=True,finished_unix=time.time()))
    mounted();temp=journal.with_suffix('.tmp');temp.write_text(json.dumps(records,indent=2)+'\n');temp.replace(journal)
    print('MIGRATED',rel,len(files),total,flush=True)
shutil.copyfile(str(journal),str(BASE/'ics_advantage_20261031_004629/manifests/nas_storage_migration.json'))
print('MIGRATION_COMPLETE',flush=True)
