"""Bulk-transfer own frozen images; remote hashes verified before local removal."""
import hashlib,json,os,shlex,shutil,subprocess,time
from pathlib import Path
BASE=Path('/data1/liuyeqiang');MOUNT=BASE/'nas/A100';STORE=MOUNT/'liuyeqiang_codex_runs'
SSH=['ssh','-p','24557','-i','/home/liuyeqiang/.ssh/id_ed25519_nas_readwrite','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=nas-readwrite','-o','UserKnownHostsFile=/home/liuyeqiang/.ssh/known_hosts_nas_readwrite','-o','ConnectTimeout=10','Valadmin@127.0.0.1']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert os.path.ismount(str(MOUNT))
journal=STORE/'cache_migration_20261031.json';records=json.loads(journal.read_text())
for rel in ['ics_evidence_20261030_215311/data/vkitti2/images','ics_advantage_20261031_004629/tsne_features']:
 src=BASE/rel;dst=STORE/rel
 if src.is_symlink():continue
 if not src.exists():continue
 files={str(p.relative_to(src)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in src.rglob('*') if p.is_file()}
 assert files
 remote='/home/Valadmin/A100/liuyeqiang_codex_runs/'+rel;stage=remote+'.bulk_staging'
 program='import os,sys,tarfile;root='+repr(stage)+';os.makedirs(root,exist_ok=True);tarfile.open(fileobj=sys.stdin.buffer,mode="r|").extractall(root)'
 tar=subprocess.Popen(['tar','-C',str(src),'-cf','-','.'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 print('BULK_BEGIN',rel,len(files),flush=True)
 with tar.stdout:
  uploaded=subprocess.run(SSH+['python3 -c '+shlex.quote(program)],stdin=tar.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
 assert tar.wait()==0,tar.stderr.read();assert uploaded.returncode==0,uploaded.stderr.decode()
 verify='''import hashlib,json,sys,os,shutil
from pathlib import Path
root=Path(sys.argv[1]);expected=json.load(sys.stdin);actual={}
for p in root.rglob('*'):
 if p.is_file():actual[str(p.relative_to(root))]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
assert actual==expected,'Remote SHA256 manifest mismatch'
dest=Path(sys.argv[2]);old=Path(str(dest)+'.incomplete_copy')
assert not old.exists()
if dest.exists():dest.rename(old)
root.rename(dest)
if old.exists():shutil.rmtree(str(old))
print(json.dumps(dict(verified=True,files=len(actual))))
'''
 check=subprocess.run(SSH+['python3 -c '+shlex.quote(verify)+' '+shlex.quote(stage)+' '+shlex.quote(remote)],input=json.dumps(files).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
 assert check.returncode==0,check.stderr.decode();print('REMOTE_HASH_PASS',check.stdout.decode().strip(),flush=True)
 assert os.path.ismount(str(MOUNT));assert len(list(dst.rglob('*')))>0
 pending=src.with_name(src.name+'.nas_verified_link');backup=src.with_name(src.name+'.local_verified_backup')
 assert not backup.exists();pending.symlink_to(dst,target_is_directory=True);src.rename(backup);os.replace(str(pending),str(src))
 shutil.rmtree(str(backup))
 records.append(dict(original=str(src),NAS=str(dst),files=len(files),bytes=sum(x['bytes'] for x in files.values()),
                     sha256_verified_every_file=True,verification='Remote NAS computes actual file hashes; exact local manifest equality',
                     path_preserved_by_symlink=True,finished_unix=time.time()))
 tmp=journal.with_suffix('.tmp');tmp.write_text(json.dumps(records,indent=2)+'\n');tmp.replace(journal)
 print('MIGRATED',rel,flush=True)
shutil.copyfile(str(journal),str(BASE/'ics_advantage_20261031_004629/manifests/nas_storage_migration.json'))
print('MIGRATION_COMPLETE',flush=True)
