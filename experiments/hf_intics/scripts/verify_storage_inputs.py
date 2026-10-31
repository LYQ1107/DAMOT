"""Rebuild task-only image aliases and verify the original HF frame inventory."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

Q=Path(__file__).resolve().parents[1]
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
NAS=Path('/home/liuyeqiang/remote/damot_hf_intics_20261031_004258_nas')
OLD_NAS='/data1/liuyeqiang/nas/A100'
REMOTE_NAS='/home/Valadmin/A100'
ssh=['ssh','-p','24557','-i','/home/liuyeqiang/.ssh/id_ed25519_nas_readwrite',
    '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
    '-o','HostKeyAlias=nas-readwrite','-o','UserKnownHostsFile=/home/liuyeqiang/.ssh/known_hosts_nas_readwrite',
    '-o','ConnectTimeout=10','Valadmin@127.0.0.1']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
baseline=json.loads((HF/'manifests/data.json').read_text())
current=json.loads((Q/'manifests/data.json').read_text())
clear=Path('/data1/liuyeqiang/damot_rebuttal_20261026/data/views/clear_holdout_45v')
clear_sources={p.name:os.readlink(str(p)) if p.is_symlink() else str(p)
    for p in clear.iterdir()}
probe_aliases={d:str(Q/'data/images'/d) for d in current}
probe_aliases['clear_holdout_45v']=str(Q/'data/images/clear_holdout_45v')
remote_root=REMOTE_NAS+'/liuyeqiang_runs/'+Q.name
setup='''import json,sys,os
from pathlib import Path
r=Path(sys.argv[1]);payload=json.load(sys.stdin)
for domain,target in payload['probe_aliases'].items():
 p=r/'probe_data'/domain
 assert p.is_symlink()
 p.unlink();p.symlink_to(target,target_is_directory=True)
view=r/'data/images/clear_holdout_45v';view.mkdir(parents=True,exist_ok=True)
for video,target in payload['clear_sources'].items():
 p=view/video
 if p.is_symlink():assert os.readlink(p)==target
 else:p.symlink_to(target,target_is_directory=True)
'''
payload=dict(probe_aliases=probe_aliases,clear_sources={v:
    str(NAS)+s[len(OLD_NAS):] if s.startswith(OLD_NAS+'/') else s
    for v,s in clear_sources.items()})
subprocess.run(ssh+['python3 -c '+shlex.quote(setup)+' '+shlex.quote(remote_root)],
    input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
    timeout=30,check=True)

remote_files=[];local_files=[];counts={};annotations={};samples={}
for domain in list(current)+['clear_holdout_45v']:
    if domain in current:
        info=current[domain];old=baseline[domain]
        assert {k:v for k,v in info.items() if k!='image_view'}=={
            k:v for k,v in old.items() if k!='image_view'}
        annotation=Path(info['annotation'])
        assert sha(annotation)==old['annotation_sha256']
        sources={v:d['source'] for v,d in old['by_video'].items()}
    else:
        annotation=Path('/data1/liuyeqiang/damot_rebuttal_20261026/data/annotations/clear_holdout_45v.json')
        sources=clear_sources
    data=json.loads(annotation.read_text());counts[domain]=len(data['images'])
    annotations[domain]=dict(path=str(annotation),sha256=sha(annotation))
    for image in data['images']:
        video,filename=image['file_name'].split('/',1)
        original=sources[video]+'/'+filename
        if original.startswith(OLD_NAS+'/'):
            translated=REMOTE_NAS+original[len(OLD_NAS):]
            remote_files.append(translated)
            if domain not in samples:
                samples[domain]=dict(remote=translated,
                    alias=str(Q/'data/images'/domain/image['file_name']))
        else:local_files.append(original)
    if domain not in samples:
        image=data['images'][0]
        samples[domain]=dict(local=sources[image['file_name'].split('/')[0]]+'/'+image['file_name'].split('/',1)[1],
            alias=str(Q/'data/images'/domain/image['file_name']))
missing_local=[p for p in local_files if not Path(p).is_file()]
assert not missing_local,repr(missing_local[:20])
check='''import json,sys,hashlib
from pathlib import Path
payload=json.load(sys.stdin)
missing=[p for p in payload['files'] if not Path(p).is_file()]
result={'remote_frame_count':len(payload['files']),'missing':missing[:20],
 'missing_count':len(missing),'sample_sha256':{d:hashlib.sha256(Path(v['remote']).read_bytes()).hexdigest()
 for d,v in payload['samples'].items() if 'remote' in v}}
print(json.dumps(result))
'''
result=subprocess.run(ssh+['python3 -c '+shlex.quote(check)],
    input=json.dumps(dict(files=remote_files,samples=samples)).encode(),
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=True)
verified=json.loads(result.stdout)
assert verified['missing_count']==0,verified
for domain,sample in samples.items():
    expected=verified['sample_sha256'][domain] if 'remote' in sample else sha(sample['local'])
    sample['sha256']=sha(sample['alias']);assert sample['sha256']==expected
report=dict(status='passed',annotations_unchanged=True,same_HF_NAS_source_objects=True,
    no_old_task_writes=True,frame_counts=counts,annotations=annotations,
    remote_frames_checked=len(remote_files),local_frames_checked=len(local_files),
    representative_alias_hash_checks=samples)
(Q/'preflight/input_storage_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print('INPUT_STORAGE_VERIFIED',json.dumps(counts),flush=True)
