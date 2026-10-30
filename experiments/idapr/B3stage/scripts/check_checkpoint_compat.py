"""Repair serialization only; preserve original trained checkpoints and every learned tensor."""
from common import *
setup('HF')
import torch
from mmcv import Config
from mmcv.runner.checkpoint import get_state_dict
from darth.models import build_model


def identical(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


records = []
for group in ['B3', 'R']:
    root = Q / 'runs' / group
    source = root / 'episode/adapt/stage04_after_night/epoch_2.pth'
    checkpoint = torch.load(str(source), map_location='cpu')
    original = checkpoint['state_dict']
    excluded = {k: v for k, v in original.items() if k.endswith('.roi_proto_space_basis')}
    assert set(excluded) == ({'track_head.roi_proto_space_basis',
                              'teacher_track_head.roi_proto_space_basis'} if group == 'B3' else set())
    expected = original.copy()
    for key in excluded:
        del expected[key]
    assert len(expected) == 752
    cfg = Config.fromfile(str(root / 'base.py'))
    model = build_model(copy.deepcopy(cfg.model))
    model.load_state_dict(expected, strict=True)
    before = model.state_dict()
    serialized = get_state_dict(model)
    assert identical(before, expected) and identical(serialized, expected)
    assert not any('roi_proto_space_basis' in k for k in serialized)
    if group == 'B3':
        for key, value in excluded.items():
            head = model.track_head if key.startswith('track_head.') else model.teacher_track_head
            assert torch.equal(value, head.roi_proto_space_basis)
    record = dict(group=group, original_checkpoint=str(source), original_sha256=sha(source),
                  original_basis_keys=list(excluded), retained_state_keys=752,
                  strict_model_load=True, pytorch_state_equal=True, mmcv_state_equal=True,
                  learned_tensors_bitwise_equal=True, no_training_or_inference_repeated=True)
    if group == 'B3':
        target = source.with_name('epoch_2_export.pth')
        payload = dict(checkpoint, state_dict=expected)
        temp = target.with_suffix('.tmp')
        torch.save(payload, str(temp))
        temp.replace(target)
        restored = torch.load(str(target), map_location='cpu')
        assert identical(restored, payload)
        record.update(exported_checkpoint=str(target), exported_sha256=sha(target),
                      all_other_checkpoint_payload_bitwise_equal=True)
        del restored, payload
    else:
        record.update(exported_checkpoint=str(source), exported_sha256=sha(source),
                      all_other_checkpoint_payload_bitwise_equal=True)
    records.append(record)
    del checkpoint, original, expected, excluded, model, before, serialized

dump(Q / 'U/checkpoint_serialization.json', dict(status='passed', records=records,
     cause='Old MMCV get_state_dict deliberately saves nonpersistent buffers; '
           'the final code adds a release-only state-dict exclusion hook.',
     original_training_audit_flag_limit='The original hook checked torch state_dict only. '
           'Its basis_not_persisted_in_checkpoint flag did not verify the MMCV file.',
     training_and_evaluation_code_preserved=True,
     numerical_training_code_unchanged=True))
print('CHECKPOINT_SERIALIZATION_PASSED', flush=True)
