"""Audit delivered numbers, immutable inputs, fixed plotting cohorts and files."""
from common import *
import ast
import subprocess
import numpy as np
from PIL import Image


def json_read(path):
    return json.loads(Path(path).read_text())


def near(actual, expected, tol=1e-10):
    assert abs(float(actual) - float(expected)) < tol, (actual, expected)


def audit_drift():
    folder = ROOT / 'drift'
    checks = read_csv(folder / 'selfchecks.csv')
    assert len(checks) == 24 and all(r['pass_check'] == 'True' for r in checks)
    rows = read_csv(folder / 'drift_restoration_summary.csv')
    replicates = read_csv(folder / 'drift_random_replicates.csv')
    assert len(rows) == 24 and len(replicates) == 60
    original = read_csv(EXPERIMENTS / 'idapr/A/A2_restoration.csv')
    original_random = read_csv(EXPERIMENTS / 'idapr/A/A2_random_replicates.csv')
    for r in rows:
        if r['basis'] == 'Random':
            rr = [x for x in replicates if x['model'] == r['model'] and x['k'] == r['k']]
            assert sorted(int(x['replicate']) for x in rr) == list(range(10))
            values = np.array([float(x['recovered_pct']) for x in rr])
            for field, value in [('recovered_pct', values.mean()), ('random_sd', values.std(ddof=0)),
                                 ('random_min', values.min()), ('random_max', values.max())]:
                near(r[field], value)
            near(r['residual_pp'], np.mean([float(x['residual_pp']) for x in rr]))
            for x in rr:
                src = [s for s in original_random if s['model'] == x['model'] and s['k'] == x['k']
                       and s['query_stage'] == '4' and s['replicate'] == x['replicate']]
                assert len(src) == 3
                near(x['recovered_pct'], 100 * sum(float(s['forgetting_reduction']) for s in src)
                     / sum(float(s['forgetting_proxy']) for s in src))
        else:
            src = [s for s in original if s['model'] == r['model'] and s['basis'] == r['basis']
                   and s['k'] == r['k'] and s['query_stage'] == '4']
            assert len(src) == 3
            near(r['recovered_pct'], 100 * sum(float(s['forgetting_reduction']) for s in src)
                 / sum(float(s['forgetting_proxy']) for s in src))
            near(r['residual_pp'], 100 * sum(float(s['forgetting_restored']) for s in src) / 3)
        near(float(r['residual_pp']), float(r['none_pp']) * (1 - float(r['recovered_pct']) / 100))
    protocol = json_read(folder / 'protocol.json')
    assert protocol['status'] == 'PASS' and protocol['no_averaging_of_domain_ratios']
    verify_sources(protocol['source_files'])
    return dict(summary_rows=len(rows), random_replicates=len(replicates), selfchecks=len(checks))


def audit_quadrants():
    folder = ROOT / 'quadrant'
    rows = read_csv(folder / 'quadrant_metrics.csv')
    assert len(rows) == 32
    counts = {}
    for model in ['F', 'HF']:
        rr = [r for r in rows if r['model'] == model]
        assert len(rr) == 16
        assert rr == read_csv(folder / ('quadrant_metrics_%s.csv' % model))
        random = [r for r in rr if r['space'] == 'Random']
        assert sorted(int(r['replicate']) for r in random) == list(range(10))
        one = {r['space']: r for r in rr if r['space'] != 'Random'}
        assert set(one) == {'INT', 'ID', 'W', 'ICS', 'Full', 'Complement'}
        for r in rr:
            for metric in ['id_mAP_euc', 'silhouette', 'within_var_ratio', 'weather_bacc', 'nightclear_bacc']:
                assert np.isfinite(float(r[metric])), (model, r['space'], metric)
            assert (r['id_mAP_cos'] != '') == (r['space'] in ['Full', 'Complement'])
            assert 0 <= float(r['within_var_ratio']) <= 1
            for field, expected in [('n_rois', 18019), ('n_test_videos', 77), ('n_videos_used', 76),
                                    ('n_silhouette_rois', 17410), ('n_variance_rois', 18019),
                                    ('n_variance_videos', 77), ('n_retrieval_queries', 4000)]:
                assert int(r[field]) == expected, (model, field, r[field])
        checks = read_csv(folder / ('selfchecks_%s.csv' % model))
        assert len(checks) == 4
        for check in checks:
            assert check['pass_check'] == 'True'
            near(check['actual'], check['reference_exact'], 1e-7)
            near(check['absolute_error'], abs(float(check['actual']) - float(check['reference_exact'])))
        gates = json_read(folder / ('decisions_%s.json' % model))
        w = np.array([float(r['weather_bacc']) for r in random])
        threshold = w.mean() + 3 * w.std(ddof=0)
        near(gates['P1']['threshold'], threshold)
        assert gates['P1']['status'] == ('PASS' if float(one['INT']['weather_bacc']) > threshold else 'FAIL')
        assert gates['P2']['status'] == ('PASS' if all(float(one['INT'][m]) < float(one['ID'][m])
                                                    for m in ['id_mAP_euc', 'silhouette']) else 'FAIL')
        for metric in ['id_mAP_euc', 'silhouette']:
            values = np.array([float(r[metric]) for r in random])
            lo, hi = values.mean() - 2 * values.std(ddof=0), values.mean() + 2 * values.std(ddof=0)
            actual = float(one['INT'][metric])
            result = 'higher' if actual > hi else 'lower' if actual < lo else 'comparable'
            assert gates['P3'][metric]['result'] == result
            near(gates['P3'][metric]['band_low'], lo)
            near(gates['P3'][metric]['band_high'], hi)
        ratio = float(one['Complement']['id_mAP_cos']) / float(one['Full']['id_mAP_cos'])
        near(gates['P4']['retention_ratio'], ratio)
        assert gates['P4']['status'] == ('PASS' if ratio >= .98 else 'FAIL')
        protocol = json_read(folder / ('metrics_protocol_%s.json' % model))
        assert protocol['status'] == 'PASS' and not protocol['basis_reestimated']
        verify_sources(protocol['source_files'])
        eligibility = json_read(folder / ('eligible_videos_%s.json' % model))
        assert len(eligibility['silhouette']) == 76 and sum(x['rois'] for x in eligibility['silhouette']) == 17410
        assert len(eligibility['variance']) == 77 and sum(x['rois'] for x in eligibility['variance']) == 18019
        plot_protocol = json_read(folder / ('figure_protocol_%s.json' % model))
        verify_sources([plot_protocol['source_protocol']])
        old_plot = json_read(plot_protocol['source_protocol']['path'])
        for key in ['quadrant_indices', 'highlighted_identities']:
            assert plot_protocol[key] == old_plot[key]
        assert plot_protocol['random_plot_replicate'] == 0
        assert not plot_protocol['points_reselected'] and not plot_protocol['bases_reestimated']
        assert plot_protocol['min_font_pt'] >= 9
        coordinates = read_csv(folder / ('plot_coordinates_%s.csv' % model))
        assert len(coordinates) == 32000
        for method in ['pca', 'tsne']:
            for space in ['INT', 'Random', 'ID', 'Complement']:
                cc = [r for r in coordinates if r['method'] == method and r['space'] == space]
                assert [int(r['data_index']) for r in cc] == old_plot['quadrant_indices']
                for r in cc:
                    assert r['model'] == model and np.isfinite([float(r['x']), float(r['y'])]).all()
                    identity = '%s|%d' % (r['video'], int(r['instance_id']))
                    index = old_plot['highlighted_identities'].index(identity) if identity in old_plot['highlighted_identities'] else -1
                    assert int(r['highlighted_identity_index']) == index
        counts[model] = dict(metric_rows=len(rr), selfchecks=len(checks), plot_coordinate_rows=len(coordinates),
                             test_rois=18019, silhouette_rois=17410, test_videos=77)
    # The numerical compatibility diagnosis must exactly reproduce all four historical weather values.
    diagnostic = read_csv(ROOT / 'logs/probe_numerical_diagnostic.csv')
    assert len(diagnostic) == 12
    replay = [r for r in diagnostic if r['MKL_runtime_threads'] == '2']
    assert len(replay) == 4 and all(float(r['difference_pp']) == 0 for r in replay)
    return counts


def audit_files():
    pngs = list((ROOT / 'drift').glob('*.png')) + list((ROOT / 'quadrant').glob('*.png'))
    assert len(pngs) == 10
    for path in pngs:
        with Image.open(str(path)) as img:
            target = 990 if '_single' in path.stem else 2100
            assert img.width == target, (path.name, img.size)
            assert all(abs(dpi - 300) < .1 for dpi in img.info['dpi'])
            img.verify()
        pdf = path.with_suffix('.pdf')
        assert pdf.is_file() and pdf.read_bytes().startswith(b'%PDF-')
    for source in (ROOT / 'scripts').glob('*.py'):
        ast.parse(source.read_text(), filename=str(source))
    subprocess.check_call(['bash', '-n', str(ROOT / 'scripts/run_all.sh')])
    for required in ['SUMMARY.md', 'README.txt', 'runtime_versions.json', 'visual_review.json',
                     'quadrant/quadrant_metrics.md', 'logs/drift.log', 'logs/quadrant_F.log', 'logs/quadrant_HF.log']:
        assert (ROOT / required).stat().st_size > 0, required
    assert json_read(ROOT / 'visual_review.json')['status'] == 'reviewed'
    # Check all tracked historical files and all untracked paths, not just source hashes.
    base = 'ed937c3'
    changed = subprocess.check_output(['git', 'diff', '--name-only', base, '--'], cwd=str(ROOT.parents[1])).decode().splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=str(ROOT.parents[1])).decode().splitlines()
    assert all(path.startswith('experiments/ics_figs_v2/') for path in changed + untracked), changed + untracked
    return dict(pngs=len(pngs), pdfs=len(pngs), all_source_syntax='PASS',
                other_experiment_directories_unchanged=True, audit_base_commit=base)


def main():
    result = dict(status='PASS', drift=audit_drift(), quadrant=audit_quadrants(), files=audit_files(),
                  cpu_only=True, all_outputs_under='experiments/ics_figs_v2/')
    write_json(ROOT / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
