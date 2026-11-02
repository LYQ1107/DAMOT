"""HF-native A2 restoration and A1 energy checks, without model inference."""
from common import *
import time
from idapr_offline import oracle_bases, image_groups, extra_views

BASES = ['Intervention-sensitive', 'Label-free nuisance', 'Random', 'Identity oracle']


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)


def task_digest(tasks):
    digest = hashlib.sha256()
    for query, gallery, positive in tasks:
        digest.update(np.asarray([query, len(gallery)], dtype=np.int64).tobytes())
        digest.update(np.asarray(gallery, dtype=np.int64).tobytes())
        digest.update(np.asarray(positive, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def domain_row(base, basis, k, value, source, reference, replicate=''):
    reduction = value - base['mAP_cross']
    residual = base['mAP_same'] - value
    denominator = base['forgetting_proxy']
    return dict(row_type='domain', model='HF', basis=basis, k=k,
                gallery_stage=base['gallery_stage'], domain=base['domain'], replicate=replicate,
                query_stage=4, queries=base['queries'], mAP_same=base['mAP_same'],
                mAP_cross=base['mAP_cross'], mAP_restored=value,
                forgetting_proxy=denominator, forgetting_reduction=reduction,
                forgetting_restored=residual, recovered_pct=100 * reduction / denominator,
                residual_pp=100 * residual, none_pp=100 * denominator,
                basis_reference=reference, source=source)


def aggregate(rows, basis, k, replicate=''):
    assert len(rows) == 3 and sorted(int(r['gallery_stage']) for r in rows) == [1, 2, 3]
    denominator = sum(float(r['forgetting_proxy']) for r in rows)
    reduction = sum(float(r['forgetting_reduction']) for r in rows)
    residual = sum(float(r['forgetting_restored']) for r in rows)
    assert abs(denominator - reduction - residual) < 1e-12
    return dict(row_type='summary', model='HF', basis=basis, k=k, gallery_stage='',
                domain='three_old_domains', replicate=replicate, query_stage=4,
                queries=sum(int(r['queries']) for r in rows),
                forgetting_proxy=denominator, forgetting_reduction=reduction,
                forgetting_restored=residual, recovered_pct=100 * reduction / denominator,
                residual_pp=100 * residual / 3, none_pp=100 * denominator / 3,
                basis_reference=rows[0]['basis_reference'], source=rows[0]['source'])


def restoration(stages, fit, native):
    old_baselines = [r for r in read_csv(IDAPR / 'A/A2_baselines.csv') if r['model'] == 'HF']
    ordinary = read_csv(IDAPR / 'A/A2_restoration.csv')
    random_source = [r for r in read_csv(IDAPR / 'A/A2_random_replicates.csv')
                     if r['model'] == 'HF' and r['query_stage'] == '4']
    records, checks, task_protocol = [], [], []
    for stage, domain in enumerate(TARGETS[:3], 1):
        mask = ~fit & (stages[1]['weather'] == domain)
        tasks = p.build_retrieval(stages[1]['video'][mask], stages[1]['frame_id'][mask],
                                  stages[1]['instance_id'][mask], 4, 30, 4000, 0)
        old, new = stages[stage]['Z'][mask], stages[4]['Z'][mask]
        same, cross = drift.cross_map(old, old, tasks), drift.cross_map(new, old, tasks)
        base = dict(gallery_stage=stage, domain=domain, queries=len(tasks),
                    mAP_same=same, mAP_cross=cross, forgetting_proxy=same - cross)
        expected = next(r for r in old_baselines if int(r['gallery_stage']) == stage)
        for key in ['mAP_same', 'mAP_cross', 'forgetting_proxy']:
            error = abs(base[key] - float(expected[key]))
            checks.append(dict(check='A2_baseline_' + key, domain=domain, k='',
                               actual=base[key], expected=float(expected[key]), error=error,
                               tolerance=1e-9, passed=error < 1e-9))
            assert error < 1e-9, checks[-1]
        assert len(tasks) == int(expected['queries'])
        task_protocol.append(dict(domain=domain, gallery_stage=stage, queries=len(tasks),
                                  test_rois=int(mask.sum()), task_sha256=task_digest(tasks), **{
                                  k: base[k] for k in ['mAP_same', 'mAP_cross', 'forgetting_proxy']}))
        for basis, original in [('Intervention-sensitive', 'FAP residual PCA'),
                                ('Identity oracle', 'Identity oracle')]:
            for k in KS:
                E = native[(original, k)]
                assert E.shape == (256, k)
                restored = new - ((new - old) @ E) @ E.T
                value = drift.cross_map(restored, old, tasks)
                records.append(domain_row(base, basis, k, value, 'HF-native recomputation', 'HF_stage1_e5 fit videos'))
                log('A2 %s %s k%d restored=%.9f' % (domain, basis, k, value))
        for k in KS:
            nuisance = next(r for r in ordinary if r['model'] == 'HF' and r['query_stage'] == '4'
                            and int(r['gallery_stage']) == stage and int(r['k']) == k
                            and r['basis'] == 'Label-free nuisance stage1')
            for field in ['mAP_same', 'mAP_cross', 'forgetting_proxy']:
                assert abs(float(nuisance[field]) - base[field]) < 1e-9
            records.append(domain_row(base, 'Label-free nuisance', k, float(nuisance['mAP_restored']),
                           'Reused idapr/A/A2_restoration.csv HF nuisance stage1', 'HF_stage1_e5 fit videos'))
            reps = [r for r in random_source if int(r['gallery_stage']) == stage and int(r['k']) == k]
            assert sorted(int(r['replicate']) for r in reps) == list(range(10))
            for r in reps:
                for field in ['mAP_same', 'mAP_cross', 'forgetting_proxy']:
                    assert abs(float(r[field]) - base[field]) < 1e-9
                record = domain_row(base, 'Random', k, float(r['mAP_restored']),
                                    'Reused idapr/A/A2_random_replicates.csv HF',
                                    'A2 seed=0 saved random bases', int(r['replicate']))
                record['row_type'] = 'random_replicate_domain'
                records.append(record)
            domain_mean = domain_row(base, 'Random', k, float(np.mean([float(r['mAP_restored']) for r in reps])),
                                     'Reused A2 ten-replicate mean', 'A2 seed=0 saved random bases')
            values = np.array([100 * float(r['forgetting_reduction']) / float(r['forgetting_proxy']) for r in reps])
            domain_mean.update(random_sd=float(values.std(ddof=0)), random_min=float(values.min()), random_max=float(values.max()))
            records.append(domain_mean)
    summaries = []
    for k in KS:
        for basis in BASES:
            if basis == 'Random':
                reps = []
                for rep in range(10):
                    rr = [r for r in records if r['row_type'] == 'random_replicate_domain'
                          and r['k'] == k and r['replicate'] == rep]
                    row = aggregate(rr, basis, k, rep)
                    row['row_type'] = 'random_replicate_summary'
                    reps.append(row)
                records.extend(reps)
                row = aggregate([r for r in records if r['row_type'] == 'domain' and r['basis'] == basis and r['k'] == k], basis, k)
                values = np.array([r['recovered_pct'] for r in reps])
                assert abs(row['recovered_pct'] - values.mean()) < 1e-10
                row.update(recovered_pct=float(values.mean()), random_sd=float(values.std(ddof=0)),
                           random_min=float(values.min()), random_max=float(values.max()))
                expected = [1.2, 5.7, 12.5][KS.index(k)]
                error = abs(row['recovered_pct'] - expected)
                checks.append(dict(check='random_recovery_pct', domain='three_old_domains', k=k,
                                   actual=row['recovered_pct'], expected=expected, error=error,
                                   tolerance=.1, passed=error < .1))
                assert error < .1, checks[-1]
            else:
                row = aggregate([r for r in records if r['row_type'] == 'domain' and r['basis'] == basis and r['k'] == k], basis, k)
            assert abs(row['none_pp'] - 1.798) < .001, row
            summaries.append(row)
    checks.append(dict(check='none_forgetting_pp', domain='three_old_domains', k='',
                       actual=summaries[0]['none_pp'], expected=1.798,
                       error=abs(summaries[0]['none_pp'] - 1.798), tolerance=.001, passed=True))
    records.extend(summaries)
    assert len(records) == 168 and len(summaries) == 12
    write_csv(ROOT / 'drift_restoration_HF_native.csv', records)
    write_csv(ROOT / 'drift_summary.csv', summaries)
    write_csv(ROOT / 'restoration_selfchecks.csv', checks)
    write_json(ROOT / 'retrieval_protocol.json', dict(model='HF', basis_reference='HF_stage1_e5',
        gap=4, window=30, max_q=4000, seed=0, category='car', query_stage=4, gallery_stages=[1, 2, 3],
        scoring='Original drift.cross_map: cosine after re-normalizing restored query and old gallery',
        restoration='new - ((new-old) @ E) @ E.T', domains=task_protocol,
        random_recomputed=False, nuisance_recomputed=False, sd_ddof=0,
        aggregation='100 * sum(forgetting_reduction) / sum(forgetting_proxy); residual_pp=100*sum(restored)/3',
        summary_raw_forgetting_columns='Sums over three domains; mAP columns are intentionally blank in summary rows',
        csv_row_types=['domain', 'summary', 'random_replicate_domain', 'random_replicate_summary']))
    log('RESTORATION_SELFCHECK_PASS %d' % len(checks))


def energy_check(data, fit, feature_root):
    car_fit = fit & (data['category'] == 1)
    car_test = ~fit & (data['category'] == 1)
    groups, grouping = image_groups(data, car_fit)
    views = {'fap': p.normalize(data['feat_fap'].astype(float))}
    views.update(extra_views(feature_root, 'HF_stage4_e5', ['photo', 'student']))
    E_id, E_n, stats = label_free_bases(data['Z'][car_fit], groups,
                                     [views[name][car_fit] for name in ['fap', 'photo', 'student']], 2, 2)
    save_basis('HF_stage4_label_free_identity_k2', E_id)
    save_basis('HF_stage4_label_free_nuisance_k2', E_n)
    E_int = np.load(str(EVIDENCE / 'part1/HF/bases/INT_00.npy'), allow_pickle=False)
    old_n = np.load(str(IDAPR / 'A/HF/bases/HF_stage4_nuisance_k2.npy'), allow_pickle=False)
    old_id = np.load(str(IDAPR / 'A/HF/bases/HF_stage4_identity_k2.npy'), allow_pickle=False)
    original = [r for r in read_csv(IDAPR / 'A/A1_subspaces.csv')
                if r['model'] == 'HF' and r['stage'] == '4' and r['k'] == '2']
    z = data['Z'][car_test]
    weather, category = data['weather'][car_test], data['category'][car_test]
    real = np.concatenate([p.shift_rows(z, weather, category, domain, DOMAINS[0]) for domain in TARGETS])
    identities = p.identity_rows(z, data['video'][car_test], data['instance_id'][car_test], category)
    result = []
    references = {'INT_00': ('nuisance', 39.39, 11.11), 'Label-free identity': ('identity', 2.55, 13.34),
                  'A1 nuisance saved': ('nuisance', 39.39, 11.11), 'A1 identity saved': ('identity', 2.55, 13.34)}
    for name, E in [('INT_00', E_int), ('Label-free identity', E_id),
                    ('A1 nuisance saved', old_n), ('A1 identity saved', old_id)]:
        assert E.shape == (256, 2) and np.allclose(E.T @ E, np.eye(2), atol=1e-9)
        kind, rounded_weather, rounded_identity = references[name]
        ref = next(r for r in original if r['kind'] == kind)
        real_share, id_share = p.energy(real, E), p.energy(identities, E)
        angles = p.principal_angles_deg(E, old_n)
        result.append(dict(model='HF', stage=4, basis=name, k=2, n_test_rois=int(car_test.sum()),
                           n_fit_rois=int(car_fit.sum()), cross_weather_pct=100 * real_share,
                           identity_pct=100 * id_share, A1_kind=kind,
                           A1_cross_weather_pct=100 * float(ref['real_cross_domain_energy_share']),
                           A1_identity_pct=100 * float(ref['identity_energy_share']),
                           cross_weather_difference_pp=100 * (real_share - float(ref['real_cross_domain_energy_share'])),
                           identity_difference_pp=100 * (id_share - float(ref['identity_energy_share'])),
                           prompt_cross_weather_pct=rounded_weather, prompt_identity_pct=rounded_identity,
                           matches_prompt_rounding=(abs(100 * real_share - rounded_weather) < .0051
                                                    and abs(100 * id_share - rounded_identity) < .0051),
                           A1_nuisance_angle_min_deg=float(angles.min()), A1_nuisance_angle_max_deg=float(angles.max()),
                           A1_nuisance_principal_angles_deg=json.dumps(angles.tolist())))
        log('ENERGY %s weather=%.9f identity=%.9f angles=%s' % (name, 100 * real_share, 100 * id_share, angles))
    write_csv(ROOT / 'S1_energy_check.csv', result)
    write_json(ROOT / 'energy_protocol.json', dict(model='HF', stage=4, split='test car RoIs',
        fit_videos=np.unique(data['video'][car_fit]).tolist(), test_videos=np.unique(data['video'][car_test]).tolist(),
        energy='sum((rows @ E)**2) / (sum(rows**2)+EPS); same p.energy as A1',
        weather_rows='Concatenate p.shift_rows for four target-domain means versus Clear; sqrt(min(n_domain,n_clear)) weights',
        identity_rows='Original p.identity_rows on test RoIs: sqrt(n_i)*(mean_i-eligible_identity_weighted_video_class_mean), min_obs=3',
        unlabeled_identity='Original label_free_bases on HF stage4 fit car RoIs; views fap/photo/student; eps_ratio=0.1; no identity or weather labels in estimator',
        image_grouping=grouping, fit_statistics=stats,
        INT_vs_recomputed_nuisance_angles_deg=p.principal_angles_deg(E_int, E_n).tolist(),
        identity_vs_A1_identity_angles_deg=p.principal_angles_deg(E_id, old_id).tolist(),
        real_shift_rows=len(real), identity_row_count=len(identities),
        policy='Saved INT_00 results are authoritative even if a reference differs; all differences retained'))


def main():
    feature_root = hf_root()
    log('HF input preflight; fail immediately if any required HF array is absent')
    inputs = feature_preflight(feature_root, [1, 2, 3, 4])
    sources = [IDAPR / 'inputs.txt', IDAPR / 'A/A2_baselines.csv', IDAPR / 'A/A2_restoration.csv',
               IDAPR / 'A/A2_random_replicates.csv', IDAPR / 'A/A1_subspaces.csv',
               IDAPR / 'A/HF/bases/HF_stage4_nuisance_k2.npy', IDAPR / 'A/HF/bases/HF_stage4_identity_k2.npy',
               EVIDENCE / 'part1/HF/bases/INT_00.npy', EVIDENCE / 'part1/HF/protocol.json',
               Path(p.__file__), Path(drift.__file__), IDAPR_SOURCE / 'idapr_offline.py', IDAPR_SOURCE / 'idapr_basis.py']
    fingerprints = inputs + [fingerprint(path) for path in sources]
    write_json(ROOT / 'input_manifest.json', dict(HF_stage1_readable=True, sources=fingerprints))
    stages = {}
    for stage in [1, 2, 3, 4]:
        stages[stage], _ = drift.load(feature_root, 'HF_stage%d_e5' % stage, stage in [1, 4])
        assert np.all(stages[stage]['category'] == 1)
    reference = stages[1]
    for stage in [2, 3, 4]:
        for key in ['video', 'instance_id', 'frame_id', 'image_id', 'category', 'bbox', 'weather']:
            assert np.array_equal(reference[key], stages[stage][key]), (stage, key)
    fit = p.split_mask(reference['video'], 0)
    old_protocol = read_json(EVIDENCE / 'part1/HF/protocol.json')
    assert sorted(np.unique(reference['video'][fit]).tolist()) == sorted(old_protocol['fit_videos'])
    assert sorted(np.unique(reference['video'][~fit]).tolist()) == sorted(old_protocol['test_videos'])
    write_json(ROOT / 'split.json', dict(fit_videos=np.unique(reference['video'][fit]).tolist(),
        test_videos=np.unique(reference['video'][~fit]).tolist(), fit_rois=int(fit.sum()),
        test_rois=int((~fit).sum()), same_as_original_HF_split=True, seed=0))
    native = oracle_bases(reference, fit)
    for (name, k), E in native.items():
        if name in ['FAP residual PCA', 'Identity oracle']:
            save_basis('HF_stage1_%s_k%d' % (name.replace(' ', '_'), k), E)
    log('HF_NATIVE_BASES_READY fit=%d test=%d' % (fit.sum(), (~fit).sum()))
    restoration(stages, fit, native)
    energy_check(stages[4], fit, feature_root)
    verify(fingerprints)
    write_json(ROOT / 'analysis_complete.json', dict(status='PASS', CPU_only=True, sources_unchanged=True,
               reference_model='HF', reference_stage=1, F_features_read=False,
               runtime_threads={name: os.environ[name] for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
               all_tasks_complete=['HF native restoration', 'S1 exact INT energy check']))
    log('HF_NATIVE_ANALYSIS_COMPLETE')


if __name__ == '__main__':
    main()
