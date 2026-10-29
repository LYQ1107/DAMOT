"""Prepare the four specified groups without consuming GPUs."""
from common import *
setup('HF')
import torch,shutil
from mmcv import Config
base=Config.fromfile(INPUTS['HF_base_config'])
snapshot=Path(INPUTS['HF_stage3_snapshot'])
archived=Path(INPUTS['HF_stage3_rank_states'][0])
state=torch.load(str(archived),map_location='cpu')
history=state['APR']['history'];assert set(history)=={'overcast','rainy','dawn_dusk'}
hp=Q/'X3/shared_apr_history.pth'
torch.save(history,str(hp))
dump(Q/'X3/apr_history_manifest.json',dict(source=str(archived),source_sha256=sha(archived),original_training_state=True,history_domains=3,history_shapes={k:{str(c):list(t.shape) for c,t in v.items()} for k,v in history.items()},shared_history=str(hp),shared_history_sha256=sha(hp),protocol='Identical original rank-0 archived prototypes for every group and rank. No reconstruction required.'))
target=Config(dict(data=dict(train=training_data(base,'night'))));target.dump(str(Q/'configs/night_target.py'))
groups={'X3-0':{},'X3-1':{'use_causal_direction_bank':False},'X3-2':{'causal_trim_alpha':0.0},'X3-3':{'causal_freeze_random_basis':True,'causal_random_basis_seed':0}}
for name,changes in groups.items():
    root=Q/'X3'/name;episode=root/'episode';(episode/'stage_snapshots').mkdir(parents=True,exist_ok=True)
    dest=episode/'stage_snapshots'/snapshot.name
    if not dest.exists():shutil.copy2(str(snapshot),str(dest))
    cfg=copy.deepcopy(base);cfg.model.track_head.roi_proto_couple_ics=False;cfg.model.track_head.update(changes)
    strip_init(cfg.model);cfg.work_dir=str(episode/'adapt/stage04_after_night');cfg.total_epochs=2
    if 'runner' in cfg:cfg.runner.max_epochs=2
    cfg.evaluation.interval=3;cfg.checkpoint_config.max_keep_ckpts=2
    cfg.custom_hooks=[copy.deepcopy(x) for x in base.custom_hooks if x['type'] not in ('APRExperimentStateHook','APRFiniteLossHook')]+[dict(type='APRFiniteLossHook',priority='HIGH'),dict(type='ICSQuickAuditHook',root=str(root),priority='LOW')]
    cfg.dump(str(root/'base.py'))
    causal={k:v for k,v in cfg.model.track_head.items() if k.startswith(('causal_','roi_proto_')) or k=='use_causal_direction_bank'}
    dump(root/'resolved_causal_roi_proto.json',causal)
    diff={k:dict(HF=base.model.track_head.get(k,'<default>'),value=v) for k,v in cfg.model.track_head.items() if base.model.track_head.get(k,'<default>')!=v}
    dump(root/'config_diff.json',dict(relative_to=INPUTS['HF_base_config'],track_head_changes=diff,bookkeeping_changes=dict(work_dir=str(cfg.work_dir),epochs_per_stage=2,custom_hooks=cfg.custom_hooks,initialization='Full snapshot loaded; init_cfg/pretrained stripped',data_paths='Read-only local annotation and image views, unchanged training transforms'),shared_apr_history_sha256=sha(hp),stage3_checkpoint_sha256=sha(dest),seed=int(base.seed),samples_per_gpu=int(base.data.samples_per_gpu),workers_per_gpu=int(base.data.workers_per_gpu)))
    print('PREPARED_X3',name,diff,flush=True)
