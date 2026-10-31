"""U1 original/new-off parity and U2 one-batch online residual angles."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

Q = Path(__file__).resolve().parents[1]
HF = Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['original', 'off', 'residual'], required=True)
a = parser.parse_args()
CODE = Q / ('code_original' if a.mode == 'original' else 'code')
sys.path.insert(0, str(CODE / 'tools/run'))
sys.path.insert(0, str(CODE))
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmtrack.datasets import build_dataloader
from darth.datasets import build_dataset
import apr_complement_experiment as hf

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
hf.install_checkpointing()
out = Q / 'preflight' / a.mode
out.mkdir(parents=True, exist_ok=True)
cfg = Config.fromfile(str(HF / 'configs/stage3.py'))
if a.mode == 'residual':
    cfg.model.track_head.update(causal_pca_mode='intervention_residual',
        causal_trim_alpha=.25, causal_loss_weight=.1, causal_min_samples=32,
        causal_covariance_shrinkage=0.)
model = hf.build_loaded(out, cfg, HF / 'episode/stage_snapshots/stage03_after_dawn_dusk.pth')
model.cuda().train()
hf.restore_rank_state(HF, model, 3, 5, final=True)
model.set_current_target_domain('night')
model.roi_proto_start_stage('night')

cache = Path('/dev/shm') / Q.name
cache.mkdir(mode=0o700, exist_ok=True)
link = Q / 'preflight/cache'
if not link.exists():
    link.symlink_to(cache, target_is_directory=True)
batch_path = cache / 'dawn_fixed_batch.pth'
rng_path = cache / 'dawn_fixed_rng.pth'
if not batch_path.exists():
    assert a.mode == 'original'
    dataset = build_dataset(copy.deepcopy(cfg.data.train))
    loader = build_dataloader(dataset, 16, 2, num_gpus=1, dist=False,
        seed=int(cfg.seed), persistent_workers=False)
    batch = next(iter(loader))
    torch.save(batch, str(batch_path))
    torch.save(dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()), str(rng_path))
else:
    batch = torch.load(str(batch_path), map_location='cpu')
state = torch.load(str(rng_path), map_location='cpu')
random.setstate(state['python'])
np.random.set_state(state['numpy'])
torch.set_rng_state(state['torch'])
torch.cuda.set_rng_state(state['cuda'])

def serial(v):
    if isinstance(v, dict):
        return {k: serial(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [serial(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    return v

prepare = model._prepare_views
def views(*args, **kwargs):
    result = prepare(*args, **kwargs)
    t, s, c = result
    fields = ['ori_shape','img_shape','pad_shape','scale_factor','flip','flip_direction']
    hf.dump(out / 'geometry.json', dict(teacher_shape=list(t[0].shape),
        student_shape=list(s[0].shape), box_coordinates_bitwise_equal=all(
            torch.equal(x,y) for x,y in zip(t[1],s[1])),
        teacher=[serial({k:m[k] for k in fields if k in m}) for m in t[5]],
        student=[serial({k:m[k] for k in fields if k in m}) for m in s[5]]))
    return result
model._prepare_views = views
residuals = []
if a.mode == 'residual':
    bank = model.track_head.causal_bank
    update = bank.update
    def capture(feats, *args, **kwargs):
        assert not feats.requires_grad
        residuals.append(feats.detach().cpu().numpy().copy())
        return update(feats, *args, **kwargs)
    bank.update = capture
parallel = MMDataParallel(model, device_ids=[0])
started = time.monotonic()
outputs = parallel.train_step(batch, None)
torch.cuda.synchronize()
elapsed = time.monotonic() - started
loss = outputs['loss'].detach().cpu()
assert torch.isfinite(loss)
torch.save(dict(loss=loss, log_vars=outputs['log_vars']), str(out / 'losses.pth'))
hf.dump(out / 'result.json', dict(mode=a.mode, loss=float(loss),
    log_vars=outputs['log_vars'], elapsed_s=elapsed, seed=int(cfg.seed),
    num_samples=outputs['num_samples'], HF_stage3_checkpoint=True,
    APR_history_domains=list(model.track_head.roi_proto_memory._history)))
if a.mode == 'off':
    original = torch.load(str(Q / 'preflight/original/losses.pth'), map_location='cpu')
    assert torch.equal(loss, original['loss']), (loss, original['loss'])
    assert outputs['log_vars'] == original['log_vars']
    hf.dump(Q / 'preflight/U1.json', dict(status='passed',
        total_loss_bitwise_equal=True, all_loss_terms_equal=True,
        original=float(original['loss']), modified=float(loss),
        HF_stage3_with_real_APR_history=True, seed=int(cfg.seed)))
if a.mode == 'residual':
    assert len(residuals) == 1
    R = residuals[0].astype(np.float64)
    np.save(str(out / 'online_residual.npy'), R)
    covariance = R.T @ R / max(len(R)-1,1)
    _, vectors = np.linalg.eigh(covariance)
    E = vectors[:,-2:][:,::-1]
    reference = np.load('/data1/liuyeqiang/idapr_20261030_153059/A/HF/bases/HF_stage3_nuisance_k2.npy')
    E = np.linalg.qr(E)[0]
    reference = np.linalg.qr(reference.astype(np.float64))[0]
    angles = np.degrees(np.arccos(np.clip(np.linalg.svd(E.T @ reference,compute_uv=False),-1.,1.)))
    actual = model.track_head.causal_bank.pcs.detach().cpu().double().numpy()
    error = float(np.max(np.abs(actual @ actual.T - np.eye(2))))
    assert error < 1e-5
    hf.dump(Q / 'preflight/U2.json', dict(status='complete_no_angle_threshold',
        principal_angles_deg=angles.tolist(), num_key_positive_rois=len(R),
        centered=False, residual_renormalized=False, bank_orthogonality_error=error,
        reference_basis_sha256=hf.sha('/data1/liuyeqiang/idapr_20261030_153059/A/HF/bases/HF_stage3_nuisance_k2.npy')))
print('PREFLIGHT_BATCH_COMPLETE', a.mode, float(loss), flush=True)
