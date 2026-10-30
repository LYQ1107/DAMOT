"""ICS-only bank reset and fixed-bank installation; strict remaining state."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mmcv.runner import get_dist_info


def file_sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_stage3(model, filename, output, logger):
    checkpoint = torch.load(filename, map_location='cpu')
    original = checkpoint['state_dict']
    target = model.state_dict()
    skipped = sorted(set(original) - set(target))
    assert not (set(target) - set(original))
    assert all(k.startswith(('track_head.causal_bank.', 'teacher_track_head.causal_bank.')) for k in skipped)
    state = {k: v for k, v in original.items() if k in target}
    reset = bool(model.track_head.causal_reset_bank_on_resume)
    adaptations = []
    reset_keys = []
    for key, value in original.items():
        if key not in target:
            continue
        is_bank = key.startswith(('track_head.causal_bank.', 'teacher_track_head.causal_bank.'))
        if reset and is_bank:
            # Constructor defaults: every bank buffer is zero except last_update_ok.
            state[key] = (torch.ones_like(target[key]) if key.endswith('.last_update_ok')
                          else torch.zeros_like(target[key]))
            reset_keys.append(key)
            adaptations.append(dict(key=key, before=list(value.shape), after=list(state[key].shape),
                                    rule='fresh constructor bank state; no stage3 PC is retained'))
        else:
            assert value.shape == target[key].shape, key
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    assert all(torch.equal(loaded[k], v) for k, v in state.items())
    assert all(torch.equal(loaded[k], v) for k, v in original.items() if k in target and k not in reset_keys)
    if reset:
        assert len(reset_keys) == 20
        for head in [model.track_head, model.teacher_track_head]:
            assert not bool(head.causal_bank.initialized.item())
            assert int(head.causal_bank.num_updates.item()) == 0
            assert not head.causal_bank.pcs.any() and not head.causal_bank.cov_bank.any()
    parameters = dict(model.named_parameters())
    assert all(torch.equal(parameters[k].detach(), original[k]) for k in parameters)
    record = dict(snapshot=str(filename), snapshot_sha256=file_sha(filename), matched_state_keys=len(state),
                  learned_parameters_bitwise_equal=True, all_nonbank_state_bitwise_equal=True,
                  bank_reset=reset, reset_keys=reset_keys, only_bank_keys_replaced=all('.causal_bank.' in k for k in reset_keys),
                  skipped_keys=skipped, only_bank_keys_skipped=all('.causal_bank.' in k for k in skipped),
                  shape_adaptations=adaptations, rank=get_dist_info()[0])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / ('warmstart_rank%d.json' % get_dist_info()[0])).write_text(json.dumps(record, indent=2))
    logger.info('[ICS_V2] strict HF stage3 warm start: %s', record)


def install_fixed_basis(head, output, logger):
    path = getattr(head, 'causal_fixed_basis_path', None)
    if path is None:
        return False
    assert head.causal_freeze_random_basis and head.causal_bank is not None
    E = np.load(path)
    assert E.shape == (head.causal_feat_dim, head.causal_num_pcs) == (256, 2)
    assert np.isfinite(E).all() and np.allclose(E.T @ E, np.eye(2), atol=1e-6)
    value = torch.from_numpy(E).T.to(head.causal_bank.pcs)
    head.causal_bank.pcs.copy_(value)
    head.causal_bank.initialized.fill_(True)
    assert torch.equal(head.causal_bank.pcs, value)
    actual = head.causal_bank.pcs.detach().cpu().double().numpy()
    orth_error = float(np.max(np.abs(actual @ actual.T - np.eye(2))))
    actual_q = np.linalg.qr(actual.T)[0]
    target_q = np.linalg.qr(E.astype(np.float64))[0]
    angles = np.degrees(np.arccos(np.clip(np.linalg.svd(actual_q.T @ target_q, compute_uv=False), -1., 1.)))
    assert orth_error < 1e-5 and float(angles.max()) < .01
    record = dict(path=str(path), sha256=file_sha(path), pcs_shape=list(actual.shape),
                  max_row_orthonormal_error=orth_error, principal_angles_deg=angles.tolist(),
                  initialized=True, num_updates=int(head.causal_bank.num_updates.item()), rank=get_dist_info()[0])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / ('fixed_basis_check_rank%d.json' % get_dist_info()[0])).write_text(json.dumps(record, indent=2))
    logger.info('[ICS_V2] fixed basis=%s sha256=%s shape=%s updates=%d',
                path, file_sha(path), E.shape, head.causal_bank.num_updates.item())
    return True
