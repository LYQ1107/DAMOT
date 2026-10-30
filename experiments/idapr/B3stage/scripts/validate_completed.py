"""Required artifact checks; no new model runs, seeds or GPU allocations."""
from common import *
setup('HF')
import numpy as np
import torch

checks = []
for name in ['U1', 'U2', 'U3', 'extractor_parity', 'warmstart', 'all_passed', 'checkpoint_serialization']:
    result = json.loads((Q / 'U' / (name + '.json')).read_text())
    assert result['status'] == 'passed'
    checks.append(dict(kind='quick_verification', name=name, passed=True))
basis_info = json.loads((Q / 'basis/E_n_k2.json').read_text())
basis = np.load(str(Q / 'basis/E_n_k2.npy'))
assert basis.shape == (256, 2)
assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-8)
assert sha(Q / 'basis/E_n_k2.npy') == basis_info['sha256']
assert sha(Q / 'shared_apr_history.pth') == 'a8c2ea74e8d0d9aed6e069f9a7cb338bd0f39599a2e9f455dc07f8a6d2cf74bc'
for result in json.loads((Q / 'manifests/code_seal.json').read_text())['files']:
    training_copy = Q / 'source_training' / result['relative_path']
    assert sha(training_copy if training_copy.exists() else Q / 'code_hf' / result['relative_path']) == result['sha256']
for result in json.loads((Q / 'manifests/final_code_seal.json').read_text())['files']:
    assert sha(Q / 'code_hf' / result['relative_path']) == result['sha256']
serialization = {r['group']: r for r in json.loads((Q / 'U/checkpoint_serialization.json').read_text())['records']}
for group in ['B3', 'R']:
    root = Q / 'runs' / group
    cfg = json.loads((root / 'resolved_causal_roi_proto.json').read_text())
    assert cfg['causal_trim_alpha'] == .1 and cfg['causal_loss_weight'] == .025
    assert cfg['causal_num_pcs'] == 2
    assert cfg['roi_proto_couple_ics'] == (group == 'R')
    assert cfg['roi_proto_space_mode'] == ('release' if group == 'B3' else 'full')
    if group == 'R':
        assert cfg['roi_proto_couple_beta'] == 1.0 and cfg['roi_proto_couple_basis'] == 'ics'
        assert cfg['roi_proto_couple_metric'] == 'projected_cosine'
    for rank in range(4):
        start = json.loads((root / ('audit_start_rank%d.json' % rank)).read_text())
        end = json.loads((root / ('audit_end_rank%d.json' % rank)).read_text())
        apr = json.loads((root / ('apr_audit_end_rank%d.json' % rank)).read_text())
        assert start['history_domains'] == end['history_domains'] == apr['history_domains'] == 3
        assert end['epoch'] == 2 and end['iteration'] == 228 and end['total_batches'] == 228
        assert apr['total_batches'] == 228 and apr['mode'] == cfg['roi_proto_space_mode']
        assert apr['coupling'] == (group == 'R')
        assert apr['fixed_basis_unchanged']
        if group == 'B3':
            assert apr['basis_sha256'] == basis_info['sha256']
        rows = list(csv.DictReader((root / ('apr_iterations_rank%d.csv' % rank)).open()))
        assert len(rows) == 228 and [int(r['iteration']) for r in rows] == list(range(1, 229))
        assert all(np.isfinite(float(r['total_loss'])) and int(r['history_domains']) == 3 for r in rows)
        apr_values = [float(r['weighted_APR_loss']) for r in rows if r['weighted_APR_loss'] != '']
        assert apr_values and np.isfinite(apr_values).all() and any(v != 0.0 for v in apr_values)
    record = serialization[group]
    assert record['learned_tensors_bitwise_equal'] and record['all_other_checkpoint_payload_bitwise_equal']
    assert sha(record['original_checkpoint']) == record['original_sha256']
    assert sha(record['exported_checkpoint']) == record['exported_sha256']
    ck = torch.load(record['exported_checkpoint'], map_location='cpu')
    assert not any('roi_proto_space_basis' in key for key in ck['state_dict'])
    del ck
    checks.append(dict(kind='formal_training', group=group, epochs=2, four_ranks=True,
                       batches_per_rank=228, history_domains=3, passed=True))
expected = {'night': (36, 7286), 'overcast': (22, 4448), 'rainy': (17, 3150), 'dawn_dusk': (33, 6476)}
for group in ['B0', 'R', 'B3', 'HF_stage3']:
    for domain in (['overcast', 'rainy', 'dawn_dusk'] if group == 'HF_stage3' else expected):
        cell = Q / 'evaluation' / group / domain
        done = json.loads((cell / 'complete.json').read_text())
        assert done['status'] == 'complete'
        assert (done['videos'], done['frames']) == expected[domain]
        rows = list(csv.DictReader((cell / 'metrics.csv').open()))
        assert len(rows) == 7 and {r['category'] for r in rows} == set(CLASSES) | {'OVERALL'}
        assert all(np.isfinite(float(r[m + '_percent'])) for r in rows for m in ['HOTA', 'DetA', 'IDF1', 'AssA'])
        checks.append(dict(kind='evaluation', group=group, domain=domain,
                           videos=done['videos'], frames=done['frames'], passed=True))
parity = list(csv.DictReader((Q / 'manifests/HF_stage3_baseline_parity.csv').open()))
assert len(parity) == 56 and all(r['passed'] == 'True' for r in parity)
warm = json.loads((Q / 'U/warmstart.json').read_text())
assert sha(INPUTS['HF_stage3_snapshot']) == warm['stage3_snapshot_sha256']
dump(Q / 'manifests/final_validation.json', dict(status='passed', checks=checks,
     single_training_seed=1790355234, formal_groups=['B3', 'R'], B0_training_reused=True,
     nuisance_basis_sha256=basis_info['sha256'], historical_stage3_parity_exact=True))
print('FINAL_VALIDATION_PASSED', len(checks), flush=True)
