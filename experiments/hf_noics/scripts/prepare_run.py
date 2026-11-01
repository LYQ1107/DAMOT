"""Two model-field changes; immutable HF code and task-only input aliases."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from mmcv import Config

Q=Path(__file__).resolve().parents[1]
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
OLD_NAS='/data1/liuyeqiang/nas/A100'
NAS=Path(json.loads((Q/'manifests/input_mount.json').read_text())['mount'])
DOMAINS=['overcast','rainy','dawn_dusk','night']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')
def difference(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        out=[]
        for k in sorted(set(a)|set(b)):
            key=path+'.'+str(k) if path else str(k)
            if k not in a or k not in b:out.append(dict(path=key,HF=a.get(k),NoICS=b.get(k)))
            else:out.extend(difference(a[k],b[k],key))
        return out
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        out=[]
        for i in range(max(len(a),len(b))):
            key=path+'[%d]'%i
            if i>=len(a) or i>=len(b):out.append(dict(path=key,HF=a[i] if i<len(a) else None,NoICS=b[i] if i<len(b) else None))
            else:out.extend(difference(a[i],b[i],key))
        return out
    return [] if a==b else [dict(path=path,HF=a,NoICS=b)]

baseline=json.loads((HF/'manifests/data.json').read_text())
data=copy.deepcopy(baseline)
aliases=[];remote_files=[];local_files=[];samples={}
for domain in DOMAINS:
    info=data[domain];annotation=Path(info['annotation'])
    assert sha(annotation)==info['annotation_sha256']
    ann=json.loads(annotation.read_text());view=Q/'data/images'/domain
    view.mkdir(parents=True,exist_ok=True)
    for video,old in info['by_video'].items():
        original=old['source']
        target=str(NAS)+original[len(OLD_NAS):] if original.startswith(OLD_NAS+'/') else original
        link=view/video
        if link.is_symlink():assert os.readlink(str(link))==target
        else:link.symlink_to(target,target_is_directory=True)
        aliases.append(dict(domain=domain,video=video,HF_source=original,NoICS_source=target,same_original_object=True))
    for im in ann['images']:
        video,filename=im['file_name'].split('/',1)
        original=info['by_video'][video]['source']+'/'+filename
        if original.startswith(OLD_NAS+'/'):
            remote='/home/Valadmin/A100'+original[len(OLD_NAS):]
            remote_files.append(remote)
            samples.setdefault(domain,dict(remote=remote,alias=str(view/im['file_name'])))
        else:
            local_files.append(original)
            samples.setdefault(domain,dict(local=original,alias=str(view/im['file_name'])))
    info['image_view']=str(view)
assert all(Path(p).is_file() for p in local_files),'Original HF local cache input missing'
ssh=['ssh','-p','24557','-i','/home/liuyeqiang/.ssh/id_ed25519_nas_readwrite','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias=nas-readwrite','-o','UserKnownHostsFile=/home/liuyeqiang/.ssh/known_hosts_nas_readwrite','-o','ConnectTimeout=10','Valadmin@127.0.0.1']
check="""import json,sys,hashlib
from pathlib import Path
p=json.load(sys.stdin);missing=[f for f in p['files'] if not Path(f).is_file()]
print(json.dumps(dict(count=len(p['files']),missing=missing[:10],missing_count=len(missing),samples={d:hashlib.sha256(Path(v['remote']).read_bytes()).hexdigest() for d,v in p['samples'].items() if 'remote' in v})))
"""
result=subprocess.run(ssh+['python3 -c '+shlex.quote(check)],input=json.dumps(dict(files=remote_files,samples=samples)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55,check=True)
verified=json.loads(result.stdout);assert verified['missing_count']==0,verified
for d,v in samples.items():
    v['sha256']=sha(v['alias']);assert v['sha256']==(verified['samples'][d] if 'remote' in v else sha(v['local']))
dump(Q/'manifests/data.json',data)
dump(Q/'manifests/input_storage_aliases.json',dict(aliases=aliases,read_only_private_input_mount=str(NAS),other_tasks_modified=False))
dump(Q/'preflight/input_storage_checks.json',dict(status='passed',annotations_unchanged=True,same_original_HF_images=True,local_frames_checked=len(local_files),NAS_frames_checked=len(remote_files),representative_hashes=samples))

model_diffs=[];bookkeeping_diffs=[];stage_checks=[]
for stage,domain in enumerate(DOMAINS,1):
    cfg=Config.fromfile(str(HF/'configs'/('stage%d.py'%stage)))
    before=copy.deepcopy(cfg._cfg_dict)
    assert int(cfg.seed)==1790355234 and cfg.total_epochs==5
    assert cfg.data.samples_per_gpu==16 and cfg.data.workers_per_gpu==2 and cfg.data.persistent_workers
    cfg.model.track_head.use_causal_direction_bank=False
    cfg.model.track_head.roi_proto_couple_ics=False
    cfg.work_dir=str(Q/'episode/adapt'/Path(cfg.work_dir).name)
    for hook in cfg.custom_hooks:
        if hook.get('type')=='APRExperimentStateHook':hook['root']=str(Q)
    cfg.custom_hooks.append(dict(type='NoICSRuntimeAuditHook',root=str(Q),stage=stage,priority='LOWEST'))
    datasets=cfg.data.train if isinstance(cfg.data.train,list) else [cfg.data.train]
    for dataset in datasets:dataset['img_prefix']=data[domain]['image_view']+'/'
    output=Q/'configs'/('stage%d.py'%stage);cfg.dump(str(output))
    actual=Config.fromfile(str(output))
    changes=difference(before,actual._cfg_dict)
    model=[dict(stage=stage,**x) for x in changes if x['path'].startswith('model.')]
    assert {x['path'] for x in model}=={'model.track_head.use_causal_direction_bank','model.track_head.roi_proto_couple_ics'},model
    assert all(x['HF'] is True and x['NoICS'] is False for x in model)
    books=[dict(stage=stage,**x) for x in changes if not x['path'].startswith('model.')]
    assert all(x['path']=='work_dir' or x['path'].startswith('custom_hooks[') or (x['path'].startswith('data.train') and x['path'].endswith('.img_prefix')) for x in books),books
    model_diffs.extend(model);bookkeeping_diffs.extend(books)
    stage_checks.append(dict(stage=stage,seed=actual.seed,config_sha256=sha(output),HF_config_sha256=sha(HF/'configs'/output.name)))
dump(Q/'config_diff.json',dict(model_changes=model_diffs,bookkeeping_changes=bookkeeping_diffs))
dump(Q/'preflight/V1_config.json',dict(status='passed',allowed_model_paths=['model.track_head.use_causal_direction_bank','model.track_head.roi_proto_couple_ics'],stages=stage_checks,bookkeeping_only_other_changes=True))
experiment=json.loads((HF/'manifests/experiment.json').read_text())
experiment.update(experiment='HF FAP + uncoupled full-space APR; ICS disabled',ICS='disabled: no causal bank and no trimmed loss',APR='unchanged HF memory/settings; original full-space cosine via coupling=False; empty stage1 history',baseline_root=str(HF),seed=1790355234,comparison='single same-seed HF pair; exactly two model fields differ',stage_gates='none',output_storage='local; read-only private NAS input aliases')
assert sha(experiment['source']['local'])==experiment['source']['sha256']
dump(Q/'manifests/experiment.json',experiment)
print('CONFIG_AND_INPUT_CHECKS_PASSED',len(model_diffs),len(bookkeeping_diffs),len(local_files),len(remote_files),flush=True)
