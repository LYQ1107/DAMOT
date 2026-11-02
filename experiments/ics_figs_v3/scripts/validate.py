"""Audit inputs, ratio-of-sums outputs, plot specifications and change scope."""
from common import *
import ast
import subprocess
from PIL import Image


def near(a, b, tolerance=1e-9):
    assert abs(float(a) - float(b)) < tolerance, (a, b, tolerance)


def main():
    manifest = read_json(ROOT / 'input_manifest.json')
    assert manifest['HF_stage1_readable']
    verify(manifest['sources'])
    features = [r for r in manifest['sources'] if 'stage' in r]
    assert len(features) == 20 and all(r['model'] == 'HF' and '/HF_stage' in r['path'] for r in features)
    assert len([r for r in features if r['stage'] == 1 and 'feat_fap_shape' in r]) == 5
    complete = read_json(ROOT / 'analysis_complete.json')
    assert complete['status'] == 'PASS' and complete['CPU_only'] and not complete['F_features_read']
    split = read_json(ROOT / 'split.json')
    assert split['fit_rois'] == 15820 and split['test_rois'] == 18019
    assert not set(split['fit_videos']) & set(split['test_videos'])
    rows = read_csv(ROOT / 'drift_restoration_HF_native.csv')
    summary = read_csv(ROOT / 'drift_summary.csv')
    assert len(rows) == 168 and len(summary) == 12
    full_summary = [r for r in rows if r['row_type'] == 'summary']
    assert len(full_summary) == len(summary)
    assert all(saved == {key: full_row[key] for key in saved}
               for saved, full_row in zip(summary, full_summary))
    checks = read_csv(ROOT / 'restoration_selfchecks.csv')
    assert len(checks) == 13 and all(r['passed'] == 'True' for r in checks)
    assert all(float(r['error']) < float(r['tolerance']) for r in checks)
    for r in summary:
        assert r['model'] == 'HF' and r['query_stage'] == '4' and int(r['queries']) == 5845
        near(r['none_pp'], 1.7982999528786014)
        rr = [d for d in rows if d['row_type'] == 'domain' and d['basis'] == r['basis'] and d['k'] == r['k']]
        assert len(rr) == 3 and sorted(int(d['gallery_stage']) for d in rr) == [1, 2, 3]
        numerator = sum(float(d['forgetting_reduction']) for d in rr)
        denominator = sum(float(d['forgetting_proxy']) for d in rr)
        near(r['recovered_pct'], 100 * numerator / denominator)
        near(r['residual_pp'], 100 * sum(float(d['forgetting_restored']) for d in rr) / 3)
        near(r['residual_pp'], float(r['none_pp']) * (1 - float(r['recovered_pct']) / 100))
        if r['basis'] == 'Random':
            reps = [d for d in rows if d['row_type'] == 'random_replicate_summary' and d['k'] == r['k']]
            assert sorted(int(d['replicate']) for d in reps) == list(range(10))
            values = []
            for rep in reps:
                rd = [d for d in rows if d['row_type'] == 'random_replicate_domain'
                      and d['k'] == r['k'] and d['replicate'] == rep['replicate']]
                assert len(rd) == 3
                value = 100 * sum(float(d['forgetting_reduction']) for d in rd) / sum(float(d['forgetting_proxy']) for d in rd)
                near(rep['recovered_pct'], value)
                values.append(value)
            values = np.asarray(values)
            for field, expected in [('recovered_pct', values.mean()), ('random_sd', values.std(ddof=0)),
                                    ('random_min', values.min()), ('random_max', values.max())]:
                near(r[field], expected)
    for path in (ROOT / 'bases').glob('*.npy'):
        E = np.load(str(path), allow_pickle=False)
        assert E.shape[0] == 256 and np.allclose(E.T @ E, np.eye(E.shape[1]), atol=1e-9)
    assert len(list((ROOT / 'bases').glob('*.npy'))) == 8
    energy = read_csv(ROOT / 'S1_energy_check.csv')
    assert len(energy) == 4
    # An actual INT/A1 difference must be disclosed, not converted into a failing job.
    for r in energy:
        near(r['cross_weather_difference_pp'], float(r['cross_weather_pct']) - float(r['A1_cross_weather_pct']))
        near(r['identity_difference_pp'], float(r['identity_pct']) - float(r['A1_identity_pct']))
        assert np.isfinite([float(r['cross_weather_pct']), float(r['identity_pct'])]).all()
        assert int(r['n_test_rois']) == 18019
    crops = read_csv(ROOT / 'crop_manifest.csv')
    original_crops = read_csv(EVIDENCE / 'part1/HF/plot_C_crops.csv')
    assert len(crops) == len(original_crops) == 20
    for new, old in zip(crops, original_crops):
        assert all(new[key] == value for key, value in old.items())
        assert new['label'] == '%s, %s' % (DISPLAY[old['weather']], ('%.3f' % float(old['e1'])).replace('-', '−'))
    crop_protocol = read_json(ROOT / 'crop_protocol.json')
    assert crop_protocol['suptitle'] is None and not crop_protocol['reselection']
    verify(crop_protocol['source_files'])
    plot = read_json(ROOT / 'drift_figure_protocol.json')
    assert plot['no_k2_annotation'] and plot['min_font_pt'] >= 8
    figures = ['drift_restoration_HF_4bars', 'drift_restoration_HF_3bars', 'intervention_extremes_HF']
    for name in figures:
        with Image.open(str(ROOT / (name + '.png'))) as img:
            assert img.width == (4200 if name.startswith('intervention') else 990)
            assert all(abs(dpi - 300) < .1 for dpi in img.info['dpi'])
            img.verify()
        assert (ROOT / (name + '.pdf')).read_bytes().startswith(b'%PDF-')
    for source in (ROOT / 'scripts').glob('*.py'):
        ast.parse(source.read_text(), filename=str(source))
    subprocess.check_call(['bash', '-n', str(ROOT / 'scripts/run_all.sh')])
    assert read_json(ROOT / 'visual_review.json')['status'] == 'reviewed'
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE_COMMIT, '--'], cwd=str(REPO)).decode().splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=str(REPO)).decode().splitlines()
    assert all(path.startswith('experiments/ics_figs_v3/') for path in changed + untracked), changed + untracked
    assert read_json(ROOT / 'runtime_versions.json')['MKL_runtime_threads'] == 4
    result = dict(status='PASS', feature_files=20, HF_stage1_clean_FAP_confirmed=True,
                  restoration_selfchecks=13, drift_csv_rows=168, drift_summary_rows=12,
                  random_summary_replicates=30, basis_files=8, saved_RoIs_preserved=20,
                  figures_PNG=3, figures_PDF=3, all_input_hashes_unchanged=True,
                  other_directories_unchanged=True, CPU_only=True, audit_base_commit=BASE_COMMIT,
                  S1_INT_matches_A1=abs(float(energy[0]['cross_weather_difference_pp'])) < 1e-8
                  and abs(float(energy[0]['identity_difference_pp'])) < 1e-8)
    write_json(ROOT / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
