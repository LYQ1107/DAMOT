import ast
import difflib
import hashlib
import json
from pathlib import Path
Q=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
experiment=json.loads((Q/'manifests/experiment.json').read_text())
source=experiment['source']
assert sha(Path(source['local']))==source['sha256']
assert experiment['seed']==1790355234
diff=json.loads((Q/'config_diff.json').read_text())
allowed={'work_dir','custom_hooks','data.train'}|{'model.track_head.'+k for k in [
    'causal_pca_mode','causal_trim_alpha','causal_loss_weight',
    'causal_min_samples','causal_covariance_shrinkage']}
assert {r['path'] for r in diff}==allowed
from mmcv import Config
import copy
config_audit=[]
for stage in range(1,5):
    before=Config.fromfile(str(Path(experiment['baseline_root'])/'configs'/('stage%d.py'%stage)))
    after=Config.fromfile(str(Q/'configs'/('stage%d.py'%stage)))
    restored=copy.deepcopy(after.data.train)
    assert len(restored)==len(before.data.train)
    for old,new in zip(before.data.train,restored):
        assert new['img_prefix'].startswith(str(Q/'data/images')+'/')
        new['img_prefix']=old['img_prefix']
    assert restored==before.data.train,'Only storage aliases may change dataset configuration'
    assert after.seed==before.seed==1790355234
    checked=copy.deepcopy(after._cfg_dict)
    checked['data']['train']=restored
    checked['work_dir']=before.work_dir
    assert len(checked['custom_hooks'])==len(before.custom_hooks)
    for old,new in zip(before.custom_hooks,checked['custom_hooks']):
        if new.get('type')=='APRExperimentStateHook':new['root']=old['root']
    for key in ['causal_pca_mode','causal_trim_alpha','causal_loss_weight',
                'causal_min_samples','causal_covariance_shrinkage']:
        checked['model']['track_head'][key]=before.model.track_head[key]
    assert checked==before._cfg_dict,'Unrequested HF setting changed'
    config_audit.append(dict(stage=stage,seed=after.seed,
        HF_config_sha256=sha(Path(experiment['baseline_root'])/'configs'/('stage%d.py'%stage)),
        INTICS_config_sha256=sha(Q/'configs'/('stage%d.py'%stage))))
(Q/'preflight/config_integrity.json').write_text(json.dumps(dict(status='passed',
    exact_HF_config_after_restoring_five_ICS_fields_and_storage_paths=True,
    stages=config_audit),indent=2)+'\n')
changes=[];baseline=[];seal=[]
for p in sorted((Q/'code').rglob('*.py')):
    if '__pycache__' in p.parts:continue
    relative=str(p.relative_to(Q/'code'));old=Q/'code_original'/relative
    ast.parse(p.read_text())
    seal.append(dict(path=relative,sha256=sha(p)))
    if old.exists():baseline.append(dict(path=relative,sha256=sha(old)))
    if not old.exists() or old.read_bytes()!=p.read_bytes():
        before=old.read_text().splitlines(True) if old.exists() else []
        changes.extend(difflib.unified_diff(before,p.read_text().splitlines(True),
            fromfile='a/'+relative if old.exists() else '/dev/null',tofile='b/'+relative))
(Q/'code_changes.patch').write_text(''.join(changes))
(Q/'manifests/code_seal.json').write_text(json.dumps(dict(files=seal),indent=2)+'\n')
(Q/'manifests/runtime_baseline.json').write_text(json.dumps(dict(
    source_root='/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420/code',
    files=baseline,source_checkpoint_sha256=source['sha256'],
    source_checkpoint_verified=True,config_changes_only_requested_ICS_and_output_paths=True),indent=2)+'\n')
print('SOURCE_SEALED',len(seal))
