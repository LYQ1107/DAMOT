"""Prepare APR-off C0-C4 under the revised protocol."""
from common import *
setup('HF')
import inspect
import shutil
import numpy as np
from mmcv import Config
from darth.models.track_heads.causal_quasi_dense_track_head import CausalQuasiDenseTrackHead

assert json.loads((Q / 'manifests/historical_complete.json').read_text())['status'] == 'complete'
base = Config.fromfile(INPUTS['HF_base_config'])
old = Config.fromfile(str(Q / 'historical/configs/SPA_oldICS/stage04_after_night.py'))
defaults = {k: v.default for k, v in inspect.signature(CausalQuasiDenseTrackHead.__init__).parameters.items()
            if k.startswith('causal_') and v.default is not inspect.Parameter.empty}
old_causal = dict(defaults, **{k: v for k, v in old.model.track_head.items() if k.startswith('causal_')})
assert old_causal['causal_pca_mode'] == 'global_pc_ema' and old_causal['causal_num_pcs'] == 1
assert old_causal['causal_trim_alpha'] == .25 and old_causal['causal_loss_weight'] == .1
assert old_causal['causal_min_samples'] == 32 and old_causal['causal_update_momentum'] == .9
nuisance = Q / 'basis/HF_stage3_A_nuisance_k2.npy'
assert sha(nuisance) == '431605fa763d13387a8e777f9bf95190b27aa62b1cc1b3256bcad26d225bc652'
random_source = Path('/data1/liuyeqiang/ics_quick_20261029_160807/X3/X3-3/expected_random_basis.npy')
random_basis = Q / 'basis/X3_seed0_random_k2.npy'
shutil.copy2(str(random_source), str(random_basis))
for p in [nuisance, random_basis]:
    E = np.load(str(p))
    assert E.shape == (256, 2) and np.allclose(E.T @ E, np.eye(2), atol=1e-6)
dump(Q / 'basis/provenance.json', dict(
    nuisance=dict(source='/data1/liuyeqiang/idapr_20261030_153059/A/HF/bases/HF_stage3_nuisance_k2.npy',
                  path=str(nuisance), sha256=sha(nuisance), estimation='HF stage3, A-stage FAP views and GT boxes; not the B3 teacher basis'),
    random=dict(source=str(random_source), path=str(random_basis), sha256=sha(random_basis),
                generation='Existing X3-3 CPU torch Generator seed0; QR of Gaussian [256,2]. No direction resampling.'),
    training_seed=1790355234))
target = Config(dict(data=dict(train=training_data(base, 'night'))))
target.dump(str(Q / 'configs/night_target.py'))
groups = {
 'C0': dict(use_causal_direction_bank=False),
 'C1': dict(old_causal, use_causal_direction_bank=True, causal_reset_bank_on_resume=True),
 'C2': dict(causal_freeze_random_basis=True, causal_fixed_basis_path=str(nuisance), causal_trim_alpha=.25, causal_loss_weight=.1),
 'C3': dict(causal_freeze_random_basis=True, causal_fixed_basis_path=str(nuisance), causal_trim_alpha=.5, causal_loss_weight=.1),
 'C4': dict(causal_freeze_random_basis=True, causal_fixed_basis_path=str(random_basis), causal_trim_alpha=.25, causal_loss_weight=.1),
}
snapshot = Path(INPUTS['HF_stage3_snapshot'])
for group, changes in groups.items():
    root = Q / 'runs' / group
    episode = root / 'episode'
    (episode / 'stage_snapshots').mkdir(parents=True, exist_ok=True)
    link = episode / 'stage_snapshots' / snapshot.name
    if not link.exists():
        link.symlink_to(snapshot)
    cfg = copy.deepcopy(base)
    cfg.model.track_head.roi_proto_couple_ics = False
    cfg.model.track_head.use_roi_proto_memory = False
    cfg.model.track_head.update(changes)
    cfg.model.track_head.causal_min_samples = 32
    strip_init(cfg.model)
    cfg.seed = 1790355234
    cfg.work_dir = str(episode / 'adapt/stage04_after_night')
    cfg.total_epochs = 2
    if 'runner' in cfg:
        cfg.runner.max_epochs = 2
    cfg.data.samples_per_gpu = 16
    cfg.data.workers_per_gpu = 2
    cfg.evaluation.interval = 3
    cfg.custom_hooks = [copy.deepcopy(h) for h in base.custom_hooks if h['type'] not in ('APRExperimentStateHook', 'APRFiniteLossHook')]
    cfg.custom_hooks += [dict(type='APRFiniteLossHook', priority='HIGH'),
                         dict(type='ICSv2AuditHook', root=str(root), priority='LOW')]
    cfg.dump(str(root / 'base.py'))
    fields = {k: v for k, v in cfg.model.track_head.items() if k.startswith(('causal_', 'roi_proto_')) or k in ['use_causal_direction_bank','use_roi_proto_memory']}
    dump(root / 'resolved_causal_roi_proto.json', fields)
    diff = {k: dict(HF=base.model.track_head.get(k, '<default>'), value=v) for k, v in cfg.model.track_head.items() if base.model.track_head.get(k, '<default>') != v}
    dump(root / 'config_diff.json', dict(relative_to=INPUTS['HF_base_config'], track_head_changes=diff,
         seed=1790355234, epochs_per_stage=2, samples_per_gpu=16, workers_per_gpu=2, gpus=4,
         APR_enabled=False, APR_history_loaded=False, stage3_checkpoint_sha256=sha(snapshot),
         causalOld_config=str(Q / 'historical/configs/SPA_oldICS/stage04_after_night.py') if group == 'C1' else None,
         checkpoint_compatibility='C1 resets all20 student/teacher causal_bank buffers to constructor state; no HF PC is cropped. All nonbank state and learned parameters restored bitwise.',
         stage3_storage='Read-only symlink to the identical HF stage3 snapshot; no duplicated input weights.'))
    print('PREPARED', group, flush=True)
