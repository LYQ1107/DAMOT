"""Validate cohort, saved-input hashes, sums, copied recovery and scope."""
from drift_energy import *
import ast
import subprocess


def near(a, b, tolerance=1e-9):
    assert abs(float(a) - float(b)) < tolerance, (a, b)


def main():
    manifest = read_json(ROOT / 'input_manifest.json')
    verify(manifest['sources'])
    assert not manifest['bases_reestimated'] and not manifest['basis_orthogonalization_applied']
    complete = read_json(ROOT / 'complete.json')
    assert complete['status'] == 'PASS' and complete['saved_basis_files'] == 36
    assert complete['CPU_only'] and complete['MKL_runtime_threads'] == 4
    rows = read_csv(ROOT / 'drift_energy_HF_native.csv')
    summaries = read_csv(ROOT / 'energy_summary.csv')
    source = read_csv(V3 / 'drift_restoration_HF_native.csv')
    source_index = {(r['row_type'], r['basis'], r['k'], r['replicate'], r['domain']): r for r in source}
    assert len(rows) == 156 and len(summaries) == 9
    for r in rows:
        assert r['model'] == 'HF' and r['query_stage'] == '4'
        energy, numerator, denominator = [float(r[k]) for k in
            ['drift_energy_pct', 'projected_energy_sum', 'total_drift_energy_sum']]
        near(energy, 100 * numerator / denominator)
        assert 0 < energy <= 100
        old = source_index[(r['row_type'], r['basis'], r['k'], r['replicate'], r['domain'])]
        near(r['recovered_forgetting_pct'], old['recovered_pct'], 1e-12)
        near(r['recovery_per_energy'], float(r['recovered_forgetting_pct']) / energy)
        near(r['theory_random_energy_pct'], 100 * int(r['k']) / 256)
        if r['row_type'] in ['summary', 'random_replicate_summary']:
            kind = 'random_replicate_domain' if r['row_type'] == 'random_replicate_summary' else 'domain'
            rr = [d for d in rows if d['row_type'] == kind and d['basis'] == r['basis']
                  and d['k'] == r['k'] and d['replicate'] == r['replicate']]
            assert len(rr) == 3 and sum(int(d['n_rois']) for d in rr) == int(r['n_rois']) == 8117
            near(numerator, sum(float(d['projected_energy_sum']) for d in rr))
            near(denominator, sum(float(d['total_drift_energy_sum']) for d in rr))
        if r['basis'] == 'Random' and r['row_type'] in ['domain', 'summary']:
            rr = [d for d in rows if d['row_type'] == 'random_replicate_' + r['row_type']
                  and d['k'] == r['k'] and d['domain'] == r['domain']]
            assert sorted(int(d['replicate']) for d in rr) == list(range(10))
            ee = np.array([float(d['drift_energy_pct']) for d in rr])
            recoveries = np.array([float(d['recovered_forgetting_pct']) for d in rr])
            ratios = np.array([float(d['recovery_per_energy']) for d in rr])
            for field, value in [('drift_energy_pct', ee.mean()), ('random_energy_sd', ee.std(ddof=0)),
                                 ('random_energy_min', ee.min()), ('random_energy_max', ee.max()),
                                 ('random_recovery_sd', recoveries.std(ddof=0)),
                                 ('random_replicate_efficiency_mean', ratios.mean()),
                                 ('random_replicate_efficiency_sd', ratios.std(ddof=0))]:
                near(r[field], value)
    for r in summaries:
        match = [d for d in rows if d['row_type'] == 'summary' and d['basis'] == r['basis'] and d['k'] == r['k']]
        assert len(match) == 1 and all(match[0][key] == value for key, value in r.items())
    cohorts = read_csv(ROOT / 'paired_cohorts.csv')
    assert [(r['domain'], int(r['gallery_stage']), int(r['n_rois'])) for r in cohorts] == [
        ('overcast', 1, 1813), ('rainy', 2, 2326), ('dawn_dusk', 3, 3978)]
    checks = read_csv(ROOT / 'random_theory_selfcheck.csv')
    assert len(checks) == 12
    combined = [r for r in checks if r['domain'] == 'three_old_domains']
    for r in combined:
        near(r['theoretical_pct'], 100 * int(r['k']) / 256)
    for path in (ROOT / 'scripts').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    subprocess.check_call(['bash', '-n', str(ROOT / 'scripts/run_all.sh')])
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE_COMMIT, '--'], cwd=str(REPO)).decode().splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=str(REPO)).decode().splitlines()
    assert all(path.startswith('experiments/ics_figs_v3_energy/') for path in changed + untracked), changed + untracked
    result = dict(status='PASS', source_hashes_unchanged=True, saved_bases=36, fitted_bases=0,
                  paired_car_test_RoIs=8117, data_rows=156, table_rows=9, random_bases_per_rank=10,
                  ratio_of_sums_checked=True, original_recovery_values_checked=True,
                  random_theory_combined_checks=[dict(k=r['k'], result=r['theoretical_within_mean_plus_minus_2SD']) for r in combined],
                  other_directories_unchanged=True, CPU_only=True, base_commit=BASE_COMMIT)
    write_json(ROOT / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
