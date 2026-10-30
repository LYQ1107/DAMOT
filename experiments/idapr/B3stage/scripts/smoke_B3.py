"""Exactly 50 requested single-GPU smoke updates, never reused for formal runs."""
from common import *
code = setup('HF')
sys.path.insert(0, str(code / 'tools/run'))
import csv
import numpy as np
import torch
from mmcv import Config
from mmtrack.apis import train_model
from mmtrack.utils import get_root_logger
from darth.datasets import build_dataset
from darth.models import build_model
from apr_complement_experiment import install_checkpointing, APRFiniteLossHook
import ics_quick_hooks
import idapr_hooks

root = Q / 'U/U3'
root.mkdir(parents=True, exist_ok=True)
cfg = Config.fromfile(str(Q / 'runs/B3/base.py'))
cfg.work_dir = str(root)
cfg.gpu_ids = [0]
cfg.seed = 1790355234
cfg.data.samples_per_gpu = 16
cfg.data.workers_per_gpu = 2
cfg.load_from = None
cfg.resume_from = None
cfg.checkpoint_config = None
cfg.workflow = [('train', 1)]
cfg.custom_hooks = [copy.deepcopy(h) for h in cfg.custom_hooks
                    if h['type'] not in ('ICSQuickAuditHook', 'APRSpaceAuditHook')]
cfg.custom_hooks += [dict(type='ICSQuickAuditHook', root=str(root), priority='LOW'),
                     dict(type='APRSpaceAuditHook', root=str(root), require_nonzero=True,
                          expected_mode='release', expected_coupling=False, priority='LOW'),
                     dict(type='StopSmokeAfterIterationsHook', iterations=50, priority='LOWEST')]
cfg.dump(str(root / 'used_config.py'))
logger = get_root_logger(log_file=str(root / 'runner.log'), log_level=cfg.log_level)
install_checkpointing()
torch.backends.cudnn.benchmark = bool(cfg.get('cudnn_benchmark', False))
seed(1790355234)
model = build_model(copy.deepcopy(cfg.model))
state = torch.load(INPUTS['HF_stage3_snapshot'], map_location='cpu')['state_dict']
model.load_state_dict(state, strict=True)
del state
dataset = build_dataset(cfg.data.train)
model.CLASSES = dataset.CLASSES
if hasattr(dataset, 'cat_name_to_label'):
    model.cat_name_to_label = dataset.cat_name_to_label
model.set_current_target_domain('night')
model.roi_proto_start_stage('night')
model.track_head.roi_proto_memory.load_state_dict_memory(
    torch.load(str(Q / 'shared_apr_history.pth'), map_location='cpu'))
assert len(model.track_head.roi_proto_memory._history) == 3
logger.info('[APR_SPACE] smoke stage4 history_domains=3 seed=1790355234')
try:
    train_model(model, [dataset], cfg, distributed=False, validate=False,
                timestamp='B3_smoke_50', meta=dict(seed=1790355234))
except idapr_hooks.SmokeComplete:
    pass
else:
    raise RuntimeError('Smoke runner did not stop at the requested 50 updates')
rows = list(csv.DictReader((root / 'apr_iterations_rank0.csv').open()))
assert len(rows) == 50 and [int(r['iteration']) for r in rows] == list(range(1, 51))
losses = np.array([float(r['weighted_APR_loss']) for r in rows])
full_reference = np.array([float(r['paired_full_reference_weighted']) for r in rows])
assert np.isfinite(losses).all() and (losses != 0).all()
assert np.isfinite(full_reference).all() and (full_reference > 0).all()
assert all(int(r['history_domains']) == 3 for r in rows)
native_log = Path('/data1/liuyeqiang/ics_quick_20261029_160807/X3/X3-0/episode/adapt/stage04_after_night/20261029_104305.log.json')
native = [json.loads(line) for line in native_log.read_text().splitlines()]
native = [r for r in native if r.get('epoch') == 1 and 0 < r.get('iter', 0) <= 50
          and 'loss_roi_proto' in r]
last = 0
weighted_sum = 0.
for row in native:
    count = int(row['iter']) - last
    weighted_sum += float(row['loss_roi_proto']) * count
    last = int(row['iter'])
assert last > 0
native_mean = weighted_sum / last
record = dict(status='passed', iterations=50, seed=1790355234, gpus=1,
              samples_per_gpu=16, workers_per_gpu=2, history_domains=3,
              all_APR_losses_finite_nonzero=True, APR_loss_mean=float(losses.mean()),
              paired_full_reference_mean=float(full_reference.mean()),
              release_to_paired_full_ratio=float(losses.mean() / full_reference.mean()),
              mean_in_share=float(np.mean([float(r['mean_in_share']) for r in rows])),
              native_B0_APR_mean=native_mean,
              release_to_native_B0_ratio=float(losses.mean() / native_mean),
              native_B0_reference_iterations=last,
              native_B0_reference_note='Existing four-GPU B0 has 20-step averages; first40 available, unpaired. No B0 training rerun.',
              paired_reference_note='Same selected features/prototypes/gate, full-space cosine loss, before release; 50 exact matching steps.',
              basis_sha256=sha(Q / 'basis/E_n_k2.npy'),
              smoke_weights_used_for_formal_training=False)
dump(root / 'complete.json', record)
dump(Q / 'U/U3.json', record)
print('U3_PASSED', json.dumps(record), flush=True)
