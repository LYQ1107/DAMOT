"""Artifact validation of the five authorized runs; no further model runs."""
from common import *
setup('HF')
import numpy as np
import torch
from mmcv import Config

groups = ['C0','C1','C2','C3','C4']
assert json.loads((Q / 'manifests/preflight.json').read_text())['status'] == 'passed'
assert not (Q / 'shared_apr_history.pth').exists()
for item in json.loads((Q / 'manifests/code_seal.json').read_text())['files']:
    assert sha(Q / 'code_hf' / item['relative_path']) == item['sha256']
checks = []
basis_info = json.loads((Q / 'basis/provenance.json').read_text())
assert sha(basis_info['nuisance']['path']) == '431605fa763d13387a8e777f9bf95190b27aa62b1cc1b3256bcad26d225bc652'
assert sha(basis_info['random']['path']) == basis_info['random']['sha256']
for group in groups:
    root = Q / 'runs' / group
    cfg = Config.fromfile(str(root / 'base.py'))
    h = cfg.model.track_head
    assert not h.use_roi_proto_memory and not h.roi_proto_couple_ics
    assert h.causal_min_samples == 32 and cfg.seed == 1790355234
    assert cfg.data.samples_per_gpu == 16 and cfg.data.workers_per_gpu == 2
    assert cfg.total_epochs == 2
    job = json.loads((Q / 'manifests' / (group + '_train_job.json')).read_text())
    assert job['status'] == 'complete' and len(job['gpus']) == 4
    assert '--apr-history' not in job['command']
    assert job['command'][job['command'].index('--seed') + 1] == '1790355234'
    snapshot = {int(v.split(',')[0]):[int(x.strip()) for x in v.split(',')[1:]]
                for v in job['prelaunch_snapshot'].splitlines()}
    assert all(all(x < 10 for x in snapshot[g]) for g in job['gpus'])
    if group == 'C1':
        assert h.causal_pca_mode == 'global_pc_ema' and h.causal_num_pcs == 1
        assert h.causal_trim_alpha == .25 and h.causal_loss_weight == .1
        assert h.causal_reset_bank_on_resume
    elif group == 'C0':
        assert not h.use_causal_direction_bank
    else:
        assert h.causal_freeze_random_basis and h.causal_num_pcs == 2
        assert h.causal_loss_weight == .1 and h.causal_trim_alpha == (.5 if group == 'C3' else .25)
    for rank in range(4):
        warm = json.loads((root / ('warmstart_rank%d.json' % rank)).read_text())
        assert warm['learned_parameters_bitwise_equal'] and warm['all_nonbank_state_bitwise_equal']
        assert warm['only_bank_keys_replaced'] and warm['only_bank_keys_skipped']
        assert warm['snapshot_sha256'] == '0dbc117207f248eae2632bffe382d38dc0e0cc67409d44986399ae67a2a9090b'
        assert len(warm['reset_keys']) == (20 if group == 'C1' else 0)
        assert len(warm['skipped_keys']) == (20 if group == 'C0' else 0)
        start = json.loads((root / ('v2_start_rank%d.json' % rank)).read_text())
        end = json.loads((root / ('v2_end_rank%d.json' % rank)).read_text())
        early = json.loads((root / ('v2_iteration50_rank%d.json' % rank)).read_text())
        assert end['epoch'] == 2 and end['iteration'] == end['total_batches'] == 228
        assert early['total_batches'] == 50
        assert not end['APR_enabled'] and not end['APR_history_loaded'] and not end['coupling']
        assert end['history_domains'] == 0 and end['min_samples'] == 32
        rows = list(csv.DictReader((root / ('iterations_rank%d.csv' % rank)).open()))
        assert len(rows) == 228 and [int(r['iteration']) for r in rows] == list(range(1,229))
        assert all(np.isfinite(float(r['total_loss'])) and r['APR_enabled']=='False' for r in rows)
        trim = [float(r['trim_loss']) for r in rows if r['trim_loss'] != '']
        if group == 'C0':
            assert not trim and end['bank_update_calls'] == 0 and not end['ICS_enabled']
        else:
            assert trim and np.isfinite(trim).all()
            assert abs(sum(trim)/len(trim)-end['loss_track_trim_mean']) < 1e-12
        for prefix in ['key','ref']:
            count = sum(r[prefix+'_trim_effective']=='True' for r in rows)
            assert count == end[prefix+'_effective_batches']
            assert abs(count/228-end[prefix+'_effective_ratio']) < 1e-12
        if group == 'C1':
            assert start['num_updates_start'] == 0 and end['num_updates'] > 0
            assert end['first_initialized_iteration'] is not None
        elif group != 'C0':
            assert end['fixed_pcs_bitwise_unchanged'] and end['bank_update_calls'] == 0
            assert end['num_updates'] == start['num_updates_start'] == 660
            basis = json.loads((root / 'episode' / ('fixed_basis_check_rank%d.json' % rank)).read_text())
            assert basis['pcs_shape'] == [2,256] and basis['max_row_orthonormal_error'] < 1e-5
            assert max(basis['principal_angles_deg']) < .01
            assert sha(basis['path']) == basis['sha256'] == end['basis_sha256']
        checks.append(dict(kind='training_rank',group=group,rank=rank,epochs=2,batches=228,passed=True))
    ck = torch.load(str(root / 'episode/adapt/stage04_after_night/epoch_2.pth'), map_location='cpu')
    state = ck['state_dict']
    assert len(state) == (732 if group == 'C0' else 752)
    if group == 'C0':
        assert not any('.causal_bank.' in key for key in state)
    elif group == 'C1':
        assert list(state['track_head.causal_bank.pcs'].shape) == [1,256]
        assert bool(state['track_head.causal_bank.initialized'].item())
    else:
        E = torch.from_numpy(np.load(h.causal_fixed_basis_path)).T.float()
        assert torch.equal(state['track_head.causal_bank.pcs'], E)
    del ck, state
expected = {'night':(36,7286),'dawn_dusk':(33,6476)}
old = Path('/data1/liuyeqiang/ics_quick_20261029_160807/X3')
for group in ['X3-0','X3-1'] + groups:
    for domain in expected:
        cell = Q / 'evaluation' / group / domain
        done = json.loads((cell / 'complete.json').read_text())
        assert done['status'] == 'complete' and done['all_frame_hota']
        assert (done['videos'],done['frames']) == expected[domain]
        values = list(csv.DictReader((cell / 'metrics.csv').open()))
        assert len(values)==7 and {r['category'] for r in values} == set(CLASSES)|{'OVERALL'}
        tracks = {r['']:r for r in csv.DictReader((cell / 'track_metrics.csv').open())}
        for row in values:
            assert all(np.isfinite(float(row[m+'_percent'])) for m in ['HOTA','DetA','IDF1','AssA'])
            assert int(row['IDSW']) == int(float(tracks[row['category']]['IDSw'])) >= 0
        if group.startswith('X3'):
            previous = {r['category']:r for r in csv.DictReader((old / group / 'evaluation' / domain / 'metrics.csv').open())}
            assert all(float(row[m+'_percent'])==float(previous[row['category']][m+'_percent'])
                       for row in values for m in ['HOTA','DetA','IDF1','AssA'])
        checks.append(dict(kind='evaluation',group=group,domain=domain,passed=True))
history = json.loads((Q / 'manifests/historical_completed.json').read_text())
assert history['status']=='complete' and history['inference_cells']==12
assert len(list(csv.DictReader((Q / 'historical_completed/stage_domain_metrics.csv').open())))==104
assert len(list(csv.DictReader((Q / 'historical_completed/old_ICS_comparisons.csv').open())))==8
dump(Q / 'manifests/final_validation.json', dict(status='passed',checks=checks,groups=groups,
     training_seed=1790355234,APR_enabled=False,APR_history_loaded=False,
     C1_bank_only_reset_verified=True,fixed_basis_and_zero_updates_verified=True,
     actual_key_ref_activation_verified=True,historical_inference_cells=12,evaluation_cells=14))
print('FINAL_VALIDATION_PASSED',len(checks),flush=True)
