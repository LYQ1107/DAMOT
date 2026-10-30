"""Artifact/protocol validation without rerunning or altering any experiment."""
import csv
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]


def read(name):
    with (W / 'A' / (name + '.csv')).open() as f:
        return list(csv.DictReader(f))


expected = dict(A1_subspaces=36, A1_estimates=54, A1_stability=12,
                A2_restoration=108, A2_summary=36, A2_baselines=6,
                A2_random_replicates=180, oracle_self_check=18)
tables = {name: read(name) for name in expected}
assert all(len(tables[name]) == count for name, count in expected.items())
for model, stage in itertools.product(['F', 'HF'], [1, 3, 4]):
    pairs = {(int(r['k_id']), int(r['k_nuis'])) for r in tables['A1_estimates']
             if r['model'] == model and int(r['stage']) == stage}
    assert pairs == set(itertools.product([2, 8, 16], repeat=2))
assert all(r['passed'] == 'True' and float(r['max_abs_delta']) < 1e-9
           for r in tables['oracle_self_check'])
for r in tables['A1_subspaces']:
    for prefix in ['identity_oracle', 'weather_oracle', 'fap_residual_pca', 'stage1_stage3']:
        values = np.asarray(json.loads(r[prefix + '_angles_deg']))
        assert len(values) == int(r[prefix + '_angle_count'])
        assert np.isfinite(values).all()
        assert abs(float(np.median(values)) - float(r[prefix + '_median_deg'])) < 1e-10
        assert abs(float(values.max()) - float(r[prefix + '_max_deg'])) < 1e-10
    for key in ['real_cross_domain_energy_share', 'identity_energy_share', 'fap_residual_energy_share']:
        assert 0 <= float(r[key]) <= 1 + 1e-10
basis_count = 0
for model in ['F', 'HF']:
    manifest = json.loads((W / 'A' / model / 'basis_manifest.json').read_text())
    assert len(manifest) == 18
    for r in manifest:
        path = W / r['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r['sha256']
        E = np.load(str(path))
        assert E.shape == (256, r['k']) and np.isfinite(E).all()
        assert np.max(np.abs(E.T @ E - np.eye(r['k']))) < 1e-8
        basis_count += 1
for model, k in itertools.product(['F', 'HF'], [2, 8, 16]):
    oracle = next(r for r in tables['A2_summary']
                  if r['model'] == model and int(r['k']) == k and r['basis'] == 'Identity oracle')
    for summary in [r for r in tables['A2_summary'] if r['model'] == model and int(r['k']) == k]:
        rows = [r for r in tables['A2_restoration'] if r['model'] == model and int(r['k']) == k
                and r['basis'] == summary['basis']]
        assert {int(r['gallery_stage']) for r in rows} == {1, 2, 3}
        ratios = [(float(r['mAP_restored']) - float(r['mAP_cross'])) /
                  (float(r['mAP_same']) - float(r['mAP_cross'])) for r in rows]
        average = float(np.mean(ratios))
        assert np.isfinite(average)
        assert abs(average - float(summary['mean_restoration_ratio'])) < 1e-12
        assert abs(average / float(oracle['mean_restoration_ratio']) -
                   float(summary['fraction_of_identity_oracle'])) < 1e-12
inputs = json.loads((W / 'manifests/feature_inputs.json').read_text())
assert len(inputs) == 40
for r in inputs:
    assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256']
base = json.loads((W / 'manifests/inputs.json').read_text())['code_base_commit']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(W / 'code'), text=True).strip() == base
assert subprocess.check_output(['git', 'diff', '--name-only'], cwd=str(W / 'code'), text=True).strip() == ''
assert json.loads((W / 'manifests/basis_math_validation.json').read_text())['status'] == 'passed'
gate = json.loads((W / 'A/gate.json').read_text())
identity_ok = all(any(r['model'] == model and int(r['k']) == 16 and
                     r['basis'] in ['Label-free identity stage1', 'Label-free identity stage3'] and
                     float(r['mean_restoration_ratio']) >= .4 and
                     float(r['fraction_of_identity_oracle']) >= .7
                     for r in tables['A2_summary']) for model in ['F', 'HF'])
nuisance_ok = all(float(r['mean_restoration_ratio']) <= .1 for r in tables['A2_summary']
                  if r['basis'] == 'Label-free nuisance stage1' and int(r['k']) == 2)
assert bool(gate['passed']) == bool(identity_ok and nuisance_ok)
record = dict(status='passed', stage_A_gate_passed=gate['passed'], row_counts=expected,
              oracle_checks=18, oracle_max_abs_delta=gate['oracle_max_abs_delta'],
              orthonormal_label_free_bases=basis_count, unchanged_feature_inputs=40,
              tracked_training_code_unchanged=True, aggregation_ratio_checks_passed=True)
(W / 'manifests/final_validation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
