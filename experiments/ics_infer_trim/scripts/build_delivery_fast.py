"""Freeze NAS files by a bulk archive; verify hashes on both sides before publishing."""
import hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STAGE=Path('/data1/liuyeqiang/ics_infer_trim_github_stage_20261031_022136')
REMOTE='/home/Valadmin/A100/liuyeqiang_codex_runs/'+ROOT.name
SSH=['ssh','-p','24557','-i','/home/liuyeqiang/.ssh/id_ed25519_nas_readwrite','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=nas-readwrite','-o','UserKnownHostsFile=/home/liuyeqiang/.ssh/known_hosts_nas_readwrite','-o','ConnectTimeout=10','Valadmin@127.0.0.1']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def remote(program,args=(),**kwargs):
    return subprocess.run(SSH+['python3 -c '+shlex.quote(program)+''.join(' '+shlex.quote(str(a)) for a in args)],check=True,**kwargs)
assert json.loads((ROOT/'manifests/validation_batch1.json').read_text())['status']=='PASS'
dest=STAGE/'delivery/experiments/ics_infer_trim';assert not dest.exists();dest.parent.mkdir(parents=True)
archive=r'''import os,sys,tarfile,hashlib,json,io
from pathlib import Path
root=Path(sys.argv[1]);manifest={};tar=tarfile.open(fileobj=sys.stdout.buffer,mode='w|')
selected=[root/'SUMMARY.md']
for folder in ['scripts','inputs','manifests','logs','analysis','jobs','results']:
 for parent,dirs,files in os.walk(str(root/folder),followlinks=False):
  dirs[:]=[d for d in dirs if d!='__pycache__']
  for name in files:
   p=Path(parent)/name
   if not p.is_file():continue
   if folder=='results' and p.suffix not in ['.csv','.py','.json']:continue
   if folder=='results' and p.suffix=='.json' and p.name not in ['evaluation_flat_metrics.json','metrics_complete.json']:continue
   if folder=='inputs' and p.suffix not in ['.npy','.json']:continue
   if folder=='scripts' and p.suffix!='.py':continue
   selected.append(p)
for p in sorted(selected):
 rel=str(p.relative_to(root));data=p.read_bytes();manifest[rel]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 info=tarfile.TarInfo('experiments/ics_infer_trim/'+rel);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('experiments/ics_infer_trim/REMOTE_SOURCE_HASHES.json');info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data));tar.close()
'''
print('BULK_DOWNLOAD_BEGIN',flush=True)
download=subprocess.Popen(SSH+['python3 -c '+shlex.quote(archive)+' '+shlex.quote(REMOTE)],stdout=subprocess.PIPE)
with download.stdout:subprocess.run(['tar','-xf','-','-C',str(STAGE/'delivery')],stdin=download.stdout,check=True)
assert download.wait()==0
manifest=json.loads((dest/'REMOTE_SOURCE_HASHES.json').read_text())
assert all(sha(dest/p)==v['sha256'] for p,v in manifest.items())
code=ROOT/'code'
for p in code.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.md','.txt','.sh','.yml','.yaml','.json']:
        q=dest/'code'/p.relative_to(code);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(str(p),str(q));assert sha(p)==sha(q)
croot=Path('/data1/liuyeqiang/ics_v2_20261030_184026')
for m in ['C0','C2','C4']:
    for d in ['night','dawn_dusk']:
        for p in (croot/'evaluation'/m/d/'evaluation/track').glob('*.csv'):
            q=dest/'results'/(m+'_b0')/d/'evaluation/track'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(str(p),str(q));assert sha(p)==sha(q)
files=sorted(p for p in dest.rglob('*') if p.is_file())
index=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
(dest/'FILE_INDEX.json').write_text(json.dumps(dict(files=index,excluded=['checkpoints','images','raw per-frame predictions and large evaluation JSON'],raw_results_retained=str(ROOT/'results')),indent=2)+'\n')
files=sorted(p for p in dest.rglob('*') if p.is_file())
(dest/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+str(p.relative_to(dest))+'\n' for p in files))
expected={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
upload=r'''import sys,os,tarfile
root=sys.argv[1];os.makedirs(root,exist_ok=False);tarfile.open(fileobj=sys.stdin.buffer,mode='r|').extractall(root)
'''
print('LOCAL_DELIVERY_COMPLETE',len(expected),flush=True)
tar=subprocess.Popen(['tar','-C',str(STAGE/'delivery'),'-cf','-','.'],stdout=subprocess.PIPE)
with tar.stdout:remote(upload,[REMOTE+'/delivery_bulk_complete'],stdin=tar.stdout)
assert tar.wait()==0
verify=r'''import json,hashlib,sys,os
from pathlib import Path
root=Path(sys.argv[1]);dest=root/'delivery_bulk_complete/experiments/ics_infer_trim';expected=json.load(sys.stdin)
actual={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
assert actual==expected,'NAS final snapshot hash mismatch'
old=root/'delivery_filewise_partial';assert not old.exists()
if (root/'delivery').exists():(root/'delivery').rename(old)
(root/'delivery_bulk_complete').rename(root/'delivery')
print(json.dumps(dict(verified=True,files=len(actual))))
'''
remote(verify,[REMOTE],input=json.dumps(expected).encode())
(ROOT/'manifests/ready_for_publication.json').write_text(json.dumps(dict(status='ready_for_review_and_publication',batch=1,files=len(expected),all_NAS_and_local_hashes_equal=True,local_staging=str(STAGE),time=time.time()),indent=2)+'\n')
(STAGE/'scripts').mkdir()
for name in ['common.py','prepare_publication.py']:shutil.copyfile(str(dest/'scripts'/name),str(STAGE/'scripts'/name))
(STAGE/'publication').symlink_to(Path('/data1/liuyeqiang/ics_infer_trim_publication_20261031_022136'),target_is_directory=True)
print('FAST_DELIVERY_READY',STAGE,flush=True)
