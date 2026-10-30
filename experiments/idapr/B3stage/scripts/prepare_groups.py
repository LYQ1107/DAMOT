"""The two requested groups, identical X3 warmstart/history/protocol."""
from common import *
setup('HF')
import shutil
import torch
from mmcv import Config

base = Config.fromfile(INPUTS['HF_base_config'])
assert base.model.track_head.roi_proto_couple_ics is True
basis = Q / 'basis/E_n_k2.npy'
assert basis.is_file()
history = Path(INPUTS['shared_apr_history'])
expected_history_sha = 'a8c2ea74e8d0d9aed6e069f9a7cb338bd0f39599a2e9f455dc07f8a6d2cf74bc'
assert sha(history) == expected_history_sha
shared = Q / 'shared_apr_history.pth'
shutil.copy2(str(history), str(shared))
assert sha(shared) == expected_history_sha
snapshot = Path(INPUTS['HF_stage3_snapshot'])
target = Config(dict(data=dict(train=training_data(base, 'night'))))
target.dump(str(Q / 'configs/night_target.py'))
groups = {
    'B3': dict(roi_proto_couple_ics=False, roi_proto_space_mode='release',
               roi_proto_space_basis_path=str(basis)),
    'R': dict(roi_proto_couple_ics=True, roi_proto_space_mode='full'),
}
for name, changes in groups.items():
    root = Q / 'runs' / name
    episode = root / 'episode'
    (episode / 'stage_snapshots').mkdir(parents=True, exist_ok=True)
    dest = episode / 'stage_snapshots' / snapshot.name
    if not dest.exists():
        shutil.copy2(str(snapshot), str(dest))
    cfg = copy.deepcopy(base)
    cfg.model.track_head.update(changes)
    strip_init(cfg.model)
    cfg.work_dir = str(episode / 'adapt/stage04_after_night')
    cfg.total_epochs = 2
    if 'runner' in cfg:
        cfg.runner.max_epochs = 2
    cfg.evaluation.interval = 3
    cfg.checkpoint_config.max_keep_ckpts = 2
    cfg.custom_hooks = [copy.deepcopy(h) for h in base.custom_hooks
                        if h['type'] not in ('APRExperimentStateHook', 'APRFiniteLossHook')]
    cfg.custom_hooks += [dict(type='APRFiniteLossHook', priority='HIGH'),
                         dict(type='ICSQuickAuditHook', root=str(root), priority='LOW'),
                         dict(type='APRSpaceAuditHook', root=str(root), expected_mode=changes['roi_proto_space_mode'],
                              expected_coupling=changes['roi_proto_couple_ics'], priority='LOW')]
    cfg.dump(str(root / 'base.py'))
    dump(root / 'resolved_causal_roi_proto.json',
         {k: v for k, v in cfg.model.track_head.items()
          if k.startswith(('causal_', 'roi_proto_')) or k == 'use_causal_direction_bank'})
    diff = {k: dict(HF=base.model.track_head.get(k, '<default>'), value=v)
            for k, v in cfg.model.track_head.items()
            if base.model.track_head.get(k, '<default>') != v}
    dump(root / 'config_diff.json', dict(relative_to=INPUTS['HF_base_config'], track_head_changes=diff,
         bookkeeping_changes=dict(work_dir=str(cfg.work_dir), epochs_per_stage=2,
         custom_hooks=cfg.custom_hooks, initialization='full archived stage3 student+teacher snapshot',
         data_paths='independent read-only views; unchanged transforms'),
         shared_apr_history_sha256=sha(shared), stage3_checkpoint_sha256=sha(dest),
         seed=1790355234, samples_per_gpu=16, workers_per_gpu=2))
    print('PREPARED', name, diff, flush=True)
