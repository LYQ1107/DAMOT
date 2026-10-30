"""CPU proof of identical full student/teacher state and three-domain history."""
from common import *
setup('HF')
import torch
from mmcv import Config
from darth.models import build_model

snapshot = Path(INPUTS['HF_stage3_snapshot'])
state = torch.load(str(snapshot), map_location='cpu')['state_dict']
history = torch.load(str(Q / 'shared_apr_history.pth'), map_location='cpu')
assert set(history) == {'overcast', 'rainy', 'dawn_dusk'}
records = []
for group in ['B3', 'R']:
    cfg = Config.fromfile(str(Q / 'runs' / group / 'base.py'))
    model = build_model(copy.deepcopy(cfg.model))
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    assert set(loaded) == set(state)
    assert all(torch.equal(value, state[key]) for key, value in loaded.items())
    assert not any('roi_proto_space_basis' in key for key in loaded)
    model.set_current_target_domain('night')
    model.roi_proto_start_stage('night')
    head = model.track_head
    head.roi_proto_memory.load_state_dict_memory(history)
    assert len(head.roi_proto_memory._history) == 3
    assert head.causal_num_pcs == 2 and head.causal_trim_alpha == .1
    assert head.causal_loss_weight == .025 and not head.causal_freeze_random_basis
    assert head.roi_proto_couple_ics == (group == 'R')
    if group == 'R':
        assert head.roi_proto_couple_beta == 1.0
        assert head.roi_proto_couple_basis == 'ics'
        assert head.roi_proto_couple_metric == 'projected_cosine'
    records.append(dict(group=group, matched_keys=len(loaded), all_tensors_bitwise_equal=True,
                        teacher_and_student_loaded=True, history_domains=3,
                        num_updates=int(head.causal_bank.num_updates.item()),
                        mode=head.roi_proto_space_mode, coupling=head.roi_proto_couple_ics,
                        basis_not_in_state_dict=True))
    del model, loaded
dump(Q / 'U/warmstart.json', dict(status='passed', records=records,
     stage3_snapshot_sha256=sha(snapshot), shared_apr_history_sha256=sha(Q / 'shared_apr_history.pth')))
print('WARMSTART_PASSED', records, flush=True)
