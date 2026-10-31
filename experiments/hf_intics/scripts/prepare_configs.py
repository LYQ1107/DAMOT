import copy
import hashlib
import json
import os
from pathlib import Path
from mmcv import Config
Q = Path(__file__).resolve().parents[1]
HF = Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
NAS = Path('/home/liuyeqiang/remote/damot_hf_intics_20261031_004258_nas')
ORIGINAL_NAS = '/data1/liuyeqiang/nas/A100'
data=json.loads((HF/'manifests/data.json').read_text())
storage_aliases=[]
for domain,info in data.items():
    view=Q/'data/images'/domain
    view.mkdir(parents=True,exist_ok=True)
    for video,source in info['by_video'].items():
        original=source['source']
        target=(str(NAS)+original[len(ORIGINAL_NAS):]
                if original.startswith(ORIGINAL_NAS+'/') else original)
        link=view/video
        if link.is_symlink():assert os.readlink(str(link))==target
        else:link.symlink_to(target,target_is_directory=True)
        storage_aliases.append(dict(domain=domain,video=video,
            HF_source=original,INTICS_source=target,same_source_object=True,
            storage_kind='NAS' if original.startswith(ORIGINAL_NAS+'/') else 'local_cache'))
    info['image_view']=str(view)

def difference(a,b,prefix=''):
    result = []
    if isinstance(a,dict) and isinstance(b,dict):
        for key in sorted(set(a)|set(b)):
            path = prefix+'.'+key if prefix else key
            if key not in a or key not in b:
                result.append(dict(path=path,HF=a.get(key),INTICS=b.get(key)))
            else:
                result += difference(a[key],b[key],path)
    elif a != b:
        result.append(dict(path=prefix,HF=a,INTICS=b))
    return result

diffs = []
for stage in range(1,5):
    cfg=Config.fromfile(str(HF/'configs'/('stage%d.py'%stage)))
    before=copy.deepcopy(cfg._cfg_dict)
    cfg.model.track_head.update(causal_pca_mode='intervention_residual',
        causal_num_pcs=2,causal_trim_alpha=.25,causal_loss_weight=.1,
        causal_min_samples=32,causal_covariance_shrinkage=0.)
    cfg.work_dir=str(Q/'episode/adapt'/Path(cfg.work_dir).name)
    for h in cfg.custom_hooks:
        if h.get('type')=='APRExperimentStateHook':h['root']=str(Q)
    datasets=cfg.data.train if isinstance(cfg.data.train,list) else [cfg.data.train]
    domain=['overcast','rainy','dawn_dusk','night'][stage-1]
    for ds in datasets:ds['img_prefix']=data[domain]['image_view']+'/'
    cfg.dump(str(Q/'configs'/('stage%d.py'%stage)))
    diffs += [dict(stage=stage,**v) for v in difference(before,cfg._cfg_dict)]
(Q/'config_diff.json').write_text(json.dumps(diffs,indent=2,default=str)+'\n')
original=json.loads((HF/'manifests/experiment.json').read_text())
original.update(experiment='HF online intervention-residual ICS',
    ICS='intervention_residual; normalized student minus normalized clean; uncentered second moment; no residual renormalization; covEMA .98; PCEMA .9 plus QR; two PCs; alpha .25; weight .1; min32; shrink0',
    baseline_root=str(HF),seed_source=str(HF/'manifests/experiment.json'),
    comparison='same-seed paired HF; only requested ICS settings and disclosed storage aliases change',
    stage1_gate='STOP if Overcast HOTA delta vs HF <=-.5 or acute bank PC1/offline fit-video FAP residual PC1 angle >45deg',
    storage='Entire new task on private NAS mount; original NAS source objects/local cache unchanged; dataset paths relocated because shared mount is absent; original HF checkpoint retention unchanged')
(Q/'manifests/experiment.json').write_text(json.dumps(original,indent=2)+'\n')
(Q/'manifests/data.json').write_text(json.dumps(data,indent=2)+'\n')
(Q/'manifests/input_storage_aliases.json').write_text(json.dumps(dict(
    no_old_task_writes=True,annotations_unchanged=True,NAS_source='Valadmin@127.0.0.1:/home/A100',
    aliases=storage_aliases),indent=2)+'\n')
print('CONFIGS_READY',len(diffs))
