"""CPU-only final APR-off startup checks; no optimizer or inference steps."""
from common import *
setup('HF')
sys.path.insert(0, str(Q / 'code_hf/tools/run'))
import logging
import numpy as np
import torch
from mmcv import Config
from mmcv.runner.checkpoint import get_state_dict
from darth.models import build_model
from ics_v2_runtime import load_stage3, install_fixed_basis

logger = logging.getLogger('ics_v2_preflight')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
assert not (Q / 'shared_apr_history.pth').exists()
records = []
for group in ['C0', 'C1', 'C2', 'C3', 'C4']:
    cfg = Config.fromfile(str(Q / 'runs' / group / 'base.py'))
    assert not cfg.model.track_head.use_roi_proto_memory
    model = build_model(copy.deepcopy(cfg.model))
    load_stage3(model, INPUTS['HF_stage3_snapshot'], Q / 'preflight' / group, logger)
    model.set_current_target_domain('night')
    head = model.track_head
    assert head.roi_proto_memory is None and not head.roi_proto_couple_ics
    assert head.causal_min_samples == 32
    fixed = install_fixed_basis(head, Q / 'preflight' / group, logger)
    bank = head.causal_bank
    record = dict(group=group, APR_enabled=False, APR_history_loaded=False, history_domains=0,
                  coupling=False, fixed=fixed, min_samples=32)
    if bank is None:
        assert group == 'C0'
    else:
        initial = bank.pcs.clone()
        start = int(bank.num_updates.item())
        if group == 'C1':
            assert start == 0 and not bool(bank.initialized.item())
            assert not bank.pcs.any() and not bank.cov_bank.any()
        features = torch.randn(128, 256, requires_grad=True)
        trimmed = bank.trim(features, alpha=head.causal_trim_alpha,
                            start_pc=head.causal_start_pc, num_pcs=head.causal_num_pcs)
        expected = features - head.causal_trim_alpha * (features @ bank.pcs.T) @ bank.pcs
        error = float((trimmed - expected).abs().max())
        assert error < 2e-6
        trimmed.square().mean().backward()
        assert torch.isfinite(features.grad).all()
        assert torch.equal(bank.pcs, initial) and int(bank.num_updates.item()) == start
        if not fixed:
            assert bank.update(features.detach()) and int(bank.num_updates.item()) == 1
            assert bool(bank.initialized.item())
        record.update(num_pcs=head.causal_num_pcs, bank_initial_updates=start,
                      fixed_bank_bitwise=True if fixed else None,
                      trim_operator_max_error=error, finite_gradient=True)
        del features, trimmed, expected
    serialized = get_state_dict(model)
    assert len(serialized) == (732 if group == 'C0' else 752)
    if fixed:
        assert torch.equal(serialized['track_head.causal_bank.pcs'], initial)
    record['serialization_state_keys'] = len(serialized)
    records.append(record)
    del model, serialized, head, bank
dump(Q / 'manifests/preflight.json', dict(status='passed', no_GPU_training_or_inference=True,
     learned_weights_warmstarted_bitwise=True, APR_disabled_all_groups=True, APR_history_not_loaded=True,
     groups=['C0','C1','C2','C3','C4'], records=records,
     stage3_snapshot_sha256=sha(INPUTS['HF_stage3_snapshot'])))
print('PREFLIGHT_PASSED_APR_OFF_C0_TO_C4', flush=True)
