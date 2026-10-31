"""Bulk-freeze the stage1 supplement and stage4-waiting status, without raw tensors."""
import datetime,hashlib,json,os,shlex,shutil,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRIM=Path('/data1/liuyeqiang/ics_infer_trim_20261031_022136')
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
STAGE=Path('/tmp')/('damot_ics_continue_'+stamp)
DELIVERY=STAGE/'delivery';DELIVERY.mkdir(parents=True)
SSH=['ssh','-p','24557','-i','/home/liuyeqiang/.ssh/id_ed25519_nas_readwrite','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=nas-readwrite','-o','UserKnownHostsFile=/home/liuyeqiang/.ssh/known_hosts_nas_readwrite','-o','ConnectTimeout=10','Valadmin@127.0.0.1']
REMOTE='/home/Valadmin/A100/liuyeqiang_codex_runs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((ROOT/'manifests/batch2_validation.json').read_text())['status']=='PASS'
assert json.loads((ROOT/'manifests/stage1_supplement_review.json').read_text())['status']=='PASS'
assert not json.loads((ROOT/'manifests/batch2_analysis_complete.json').read_text())['stage4_available']
program=r'''import io,sys,os,json,hashlib,tarfile,time
from pathlib import Path
base=Path(sys.argv[1]);tar=tarfile.open(fileobj=sys.stdout.buffer,mode='w|');manifest={}
def add(p,name):
 for attempt in range(20):
  try:data=p.read_bytes();break
  except FileNotFoundError:
   if attempt==19:raise
   time.sleep(.1)
 info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
 manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
adv=base/'ics_advantage_20261031_004629'
for folder in ['analysis','tracking','figures','manifests','logs']:
 for parent,dirs,files in os.walk(str(adv/folder),followlinks=False):
  dirs[:]=[d for d in dirs if d!='__pycache__']
  for name in sorted(files):
   p=Path(parent)/name
   if p.suffix in ['.npz','.pth','.pkl','.pyc'] or p.name=='delivery_index.json':continue
   add(p,'experiments/ics_advantage/'+str(p.relative_to(adv)))
for folder in ['features','tsne_features','vk_features','smoke_features']:
 for p in sorted((adv/folder).rglob('*meta.json')):add(p,'experiments/ics_advantage/'+str(p.relative_to(adv)))
trim=base/'ics_infer_trim_20261031_022136';frozen=trim/'delivery/experiments/ics_infer_trim'
updates=['SUMMARY.md','scripts/analysis.py','scripts/batch2_controller.py','logs/batch2_controller.log',
 'manifests/batch2_status.json','manifests/batch2_controller_process.json','manifests/scope_decision_20261031.json','manifests/source_readiness_context.json']
for p in sorted(frozen.rglob('*')):
 if p.is_file() and str(p.relative_to(frozen)) not in updates:add(p,'experiments/ics_infer_trim/'+str(p.relative_to(frozen)))
for rel in updates:add(trim/rel,'experiments/ics_infer_trim/'+rel)
data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('REMOTE_TRANSFER_HASHES.json');info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data));tar.close()
'''
print('BULK_FREEZE_BEGIN',STAGE,flush=True)
download=subprocess.Popen(SSH+['python3 -c '+shlex.quote(program)+' '+shlex.quote(REMOTE)],stdout=subprocess.PIPE)
with download.stdout:subprocess.run(['tar','-xf','-','-C',str(DELIVERY)],stdin=download.stdout,check=True)
assert download.wait()==0
remote_manifest=json.loads((DELIVERY/'REMOTE_TRANSFER_HASHES.json').read_text())
assert all(sha(DELIVERY/p)==r['sha256'] for p,r in remote_manifest.items())
adv=DELIVERY/'experiments/ics_advantage';adv.mkdir(parents=True,exist_ok=True)
local=[ROOT/'SUMMARY.md',ROOT/'README.txt']+list((ROOT/'scripts').glob('*.py'))
sources=[]
for name in ['code','code_online']:
 for p in (ROOT/name).rglob('*.py'):
  if '__pycache__' not in p.parts:
   local.append(p);sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
local+=list((ROOT/'inputs/model_bases').glob('*.npy'))
for p in local:
 q=adv/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(str(p),str(q));assert sha(q)==sha(p)
(adv/'manifests/model_source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
advfiles=sorted(p for p in adv.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and p!=adv/'manifests/delivery_index.json')
payload=[dict(path=str(p.relative_to(DELIVERY)),bytes=p.stat().st_size,sha256=sha(p)) for p in advfiles]
index=adv/'manifests/delivery_index.json';index.write_text(json.dumps(payload,indent=2)+'\n')
checks=sorted(payload+[dict(path=str(index.relative_to(DELIVERY)),sha256=sha(index))],key=lambda r:r['path'])
(adv/'SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in checks))
trim=DELIVERY/'experiments/ics_infer_trim';remote_hash=trim/'REMOTE_SOURCE_HASHES.json'
previous=json.loads(remote_hash.read_text())
for rel in list(previous)+['manifests/scope_decision_20261031.json','manifests/source_readiness_context.json']:
 p=trim/rel
 if p.exists():previous[rel]=dict(bytes=p.stat().st_size,sha256=sha(p))
remote_hash.write_text(json.dumps(previous,indent=2)+'\n')
trimfiles=sorted(p for p in trim.rglob('*') if p.is_file() and p.name not in ['FILE_INDEX.json','SHA256SUMS.txt'])
index=json.loads((trim/'FILE_INDEX.json').read_text())
index['files']=[dict(path=str(p.relative_to(trim)),bytes=p.stat().st_size,sha256=sha(p)) for p in trimfiles]
(trim/'FILE_INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
alltrim=sorted(p for p in trim.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
(trim/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+str(p.relative_to(trim))+'\n' for p in alltrim))
record=dict(stage=str(STAGE),delivery=str(DELIVERY),files={name:len([p for p in (DELIVERY/'experiments'/name).rglob('*') if p.is_file()]) for name in ['ics_advantage','ics_infer_trim']},all_source_transfer_hashes_verified=True,stage4_main_comparison_pending=True)
(ROOT/'manifests/continuation_delivery.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
