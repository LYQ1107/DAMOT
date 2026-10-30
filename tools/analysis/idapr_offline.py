"""Gated idAPR Stage A; exact S3 retrieval and independently fitted car bases.

Run --model F and --model HF independently, then --aggregate. No GPU or model
training dependencies are imported. No labels enter label_free_bases.
"""
import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import ics_probe_analyze as p
import ics_probe_drift as drift
from idapr_basis import label_free_bases

KS = (2, 8, 16)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def progress(root, phase, **kwargs):
    record = dict(phase=phase, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    dump(root / 'progress.json', record)
    print(json.dumps(record), flush=True)


def angles(A, B):
    a = p.principal_angles_deg(A, B)
    return dict(angles_deg=json.dumps(a.tolist()), median_deg=float(np.median(a)),
                max_deg=float(np.max(a)), angle_count=len(a))


def oracle_bases(reference, fit):
    z = reference['Z']
    ident = p.identity_rows(z[fit], reference['video'][fit],
                            reference['instance_id'][fit], reference['category'][fit])
    weather = p.between_weather_rows(z[fit], reference['weather'][fit],
                                     reference['category'][fit])
    residual = p.normalize(reference['feat_fap'].astype(float)) - z
    return {(name, k): p.rank_basis(rows, k, center)
            for name, rows, center in [('Identity oracle', ident, False),
                                       ('Weather oracle', weather, False),
                                       ('FAP residual PCA', residual[fit], True)]
            for k in KS}


def image_groups(data, mask):
    """Namespace IDs only if an NPZ ID denotes multiple physical images."""
    ids = data['image_id'][mask]
    physical = np.array(['%s|%s|%s' % (w, v, f) for w, v, f in
                         zip(data['weather'][mask], data['video'][mask], data['frame_id'][mask])])
    pairs = set(zip(ids.tolist(), physical.tolist()))
    collision = len(pairs) != len(np.unique(ids))
    return (physical if collision else ids), dict(
        raw_image_ids=int(len(np.unique(ids))), physical_images=int(len(np.unique(physical))),
        image_id_namespaced=collision)


def extra_views(feat_root, label, names):
    result = {}
    for name in names:
        arrays = []
        for domain in drift.DOMAINS:
            with np.load(str(feat_root / label / (domain + '.npz')), allow_pickle=False) as z:
                arrays.append(z['feat_' + name])
        result[name] = p.normalize(np.concatenate(arrays).astype(float))
    return result


def run_model(work, model):
    root = work / 'A' / model
    root.mkdir(parents=True, exist_ok=True)
    (root / 'bases').mkdir(exist_ok=True)
    inputs = json.loads((work / 'manifests/inputs.json').read_text())
    ref, _ = drift.load(Path(inputs['F_feat_root']), 'F_stage1_e5', True)
    fit = p.split_mask(ref['video'], 0)
    test = ~fit
    oracle = oracle_bases(ref, fit)
    for (name, k), E in oracle.items():
        np.save(str(root / 'bases' / ('oracle_%s_k%d.npy' % (name.replace(' ', '_'), k))), E)
    stages = {}
    feat_root = Path(inputs[model + '_feat_root'])
    for stage in range(1, 5):
        data, _ = drift.load(feat_root, '%s_stage%d_e5' % (model, stage), stage in (1, 3, 4))
        for key in ['video', 'instance_id', 'frame_id', 'image_id', 'category', 'bbox', 'weather']:
            if not np.array_equal(ref[key], data[key]):
                raise AssertionError((model, stage, 'ROI metadata mismatch', key))
        stages[stage] = data
    progress(root, 'inputs_loaded', model=model, rois=len(ref['Z']),
             fit_rois=int(fit.sum()), test_rois=int(test.sum()))

    bases = {}
    stats = {}
    basis_manifest = []
    for stage in (1, 3, 4):
        data = stages[stage]
        car_fit = fit & (data['category'] == 1)
        image_id, grouping = image_groups(data, car_fit)
        names = ['fap'] if stage != 4 else ['fap', 'photo', 'student']
        views = {'fap': p.normalize(data['feat_fap'].astype(float))}
        if stage == 4:
            views.update(extra_views(feat_root, '%s_stage4_e5' % model, ['photo', 'student']))
        for k in KS:
            eid, en, st = label_free_bases(data['Z'][car_fit], image_id,
                                          [views[name][car_fit] for name in names], k, k)
            stats[(stage, k)] = dict(st, **grouping)
            for kind, E in [('identity', eid), ('nuisance', en)]:
                bases[(stage, kind, k)] = E
                path = root / 'bases' / ('%s_stage%d_%s_k%d.npy' % (model, stage, kind, k))
                np.save(str(path), E)
                basis_manifest.append(dict(model=model, stage=stage, kind=kind, k=k,
                    views=names, path=str(path.relative_to(work)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **stats[(stage, k)]))
        progress(root, 'A1_basis_fit', model=model, stage=stage, views=names,
                 fit_car_rois=int(car_fit.sum()), grouping=grouping)
    dump(root / 'basis_manifest.json', basis_manifest)

    stability = []
    stability_index = {}
    for kind in ['identity', 'nuisance']:
        for k in KS:
            a = angles(bases[(1, kind, k)], bases[(3, kind, k)])
            stability_index[(kind, k)] = a
            stability.append(dict(model=model, kind=kind, k=k, **a))
    p.write_csv(str(root / 'A1_stability.csv'), stability)
    properties = []
    for stage in (1, 3, 4):
        data = stages[stage]
        mask = test & (data['category'] == 1)
        z = data['Z'][mask]
        car_cls, weather = data['category'][mask], data['weather'][mask]
        # P1's actual reference-domain protocol: weighted mean shifts vs Clear.
        real = np.concatenate([p.shift_rows(z, weather, car_cls, d, 'clear_holdout_45v')
                               for d in drift.TARGETS], axis=0)
        between = p.between_weather_rows(z, weather, car_cls)
        ident = p.identity_rows(z, data['video'][mask], data['instance_id'][mask], car_cls)
        residual = p.normalize(data['feat_fap'][mask].astype(float)) - z
        for kind in ['identity', 'nuisance']:
            for k in KS:
                E = bases[(stage, kind, k)]
                row = dict(model=model, stage=stage, kind=kind, k=k,
                           views='fap' if stage != 4 else 'fap,photo,student',
                           n_test_car_rois=int(mask.sum()), **stats[(stage, k)])
                for name in ['Identity oracle', 'Weather oracle', 'FAP residual PCA']:
                    prefix = name.lower().replace(' ', '_')
                    row.update({prefix + '_' + key: val
                                for key, val in angles(E, oracle[(name, k)]).items()})
                row.update(real_cross_domain_energy_share=p.energy(real, E),
                           between_weather_energy_share=p.energy(between, E),
                           identity_energy_share=p.energy(ident, E),
                           fap_residual_energy_share=p.energy(residual, E))
                row.update({'stage1_stage3_' + key: val
                            for key, val in stability_index[(kind, k)].items()})
                properties.append(row)
    p.write_csv(str(root / 'A1_subspaces.csv'), properties)
    # The two dimensions are independent. Retain all nine requested combinations
    # per feature setting, using the same separately fitted component spaces.
    prop_index = {(r['stage'], r['kind'], r['k']): r for r in properties}
    combinations = []
    for stage, kid, knuis in itertools.product((1, 3, 4), KS, KS):
        row = dict(model=model, stage=stage, k_id=kid, k_nuis=knuis)
        for kind, k in [('identity', kid), ('nuisance', knuis)]:
            row.update({kind + '_' + key: value for key, value in
                        prop_index[(stage, kind, k)].items()
                        if key not in ['model', 'stage', 'kind', 'k']})
        combinations.append(row)
    p.write_csv(str(root / 'A1_estimates.csv'), combinations)

    historical = [r for r in read_csv(inputs['historical_restoration_csv'])
                  if r['model'] == model and int(r['query_stage']) == 4 and
                  r['basis'] == 'Identity oracle']
    assert len(historical) == 9, (model, len(historical))
    lookup = {(int(r['gallery_stage']), int(r['k_requested'])): r for r in historical}
    contexts = []
    baselines = []
    rows = []
    checks = []

    def restored_row(base, name, k, value, **kwargs):
        denom = base['mAP_same'] - base['mAP_cross']
        if denom == 0:
            raise ValueError('Undefined restoration ratio: zero forgetting')
        return dict(base, basis=name, k=k, mAP_restored=float(value),
                    forgetting_restored=base['mAP_same'] - value,
                    forgetting_reduction=value - base['mAP_cross'],
                    restoration_ratio=(value - base['mAP_cross']) / denom, **kwargs)

    # Mandatory oracle regression is completed before testing new restorations.
    for j in (1, 2, 3):
        domain = drift.TARGETS[j - 1]
        mask = test & (ref['weather'] == domain)
        tasks = p.build_retrieval(ref['video'][mask], ref['frame_id'][mask],
                                  ref['instance_id'][mask], 4, 30, 4000, 0)
        old, new = stages[j]['Z'][mask], stages[4]['Z'][mask]
        same = drift.cross_map(old, old, tasks)
        cross = drift.cross_map(new, old, tasks)
        base = dict(model=model, query_stage=4, gallery_stage=j, domain=domain,
                    queries=len(tasks), mAP_same=same, mAP_cross=cross,
                    forgetting_proxy=same - cross, seed=0, gap=4, window=30, max_q=4000)
        baselines.append(base)
        contexts.append((j, old, new, tasks, base))
        for k in KS:
            E = oracle[('Identity oracle', k)]
            fixed = new - ((new - old) @ E) @ E.T
            value = drift.cross_map(fixed, old, tasks)
            row = restored_row(base, 'Identity oracle', k, value, k_effective=E.shape[1])
            rows.append(row)
            expected = lookup[(j, k)]
            fields = ['mAP_same', 'mAP_cross', 'forgetting_proxy', 'mAP_restored',
                      'forgetting_restored', 'forgetting_reduction']
            deltas = {key: abs(float(row[key]) - float(expected[key])) for key in fields}
            checks.append(dict(model=model, gallery_stage=j, domain=domain, k=k,
                               queries=len(tasks), expected_queries=int(expected['queries']),
                               max_abs_delta=max(deltas.values()), passed=
                               len(tasks) == int(expected['queries']) and max(deltas.values()) < 1e-9,
                               **{key + '_abs_delta': value for key, value in deltas.items()}))
        p.write_csv(str(root / 'oracle_self_check.csv'), checks)
        progress(root, 'oracle_self_check', model=model, gallery_stage=j,
                 checks=len(checks), all_passed=all(r['passed'] for r in checks))
    if not all(r['passed'] for r in checks):
        raise AssertionError('Oracle regression failed; do not evaluate the gate')
    p.write_csv(str(root / 'A2_baselines.csv'), baselines)
    p.write_csv(str(root / 'A2_restoration.csv'), rows)

    rng = np.random.RandomState(0)
    random = {k: [p.orth(rng.randn(ref['Z'].shape[1], k)) for _ in range(10)] for k in KS}
    for k in KS:
        for rep, E in enumerate(random[k]):
            np.save(str(root / 'bases' / ('random_k%d_rep%d.npy' % (k, rep))), E)
    random_rows = []
    for j, old, new, tasks, base in contexts:
        options = [
            ('Label-free identity stage1', lambda k: bases[(1, 'identity', k)]),
            ('Label-free identity stage3', lambda k: bases[(3, 'identity', k)]),
            ('Label-free nuisance stage1', lambda k: bases[(1, 'nuisance', k)]),
            ('FAP residual PCA', lambda k: oracle[('FAP residual PCA', k)]),
        ]
        for name, select in options:
            for k in KS:
                E = select(k)
                fixed = new - ((new - old) @ E) @ E.T
                value = drift.cross_map(fixed, old, tasks)
                rows.append(restored_row(base, name, k, value, k_effective=E.shape[1]))
            p.write_csv(str(root / 'A2_restoration.csv'), rows)
            progress(root, 'A2_restoration', model=model, gallery_stage=j,
                     basis=name, completed_rows=len(rows))
        for k in KS:
            values = []
            for rep, E in enumerate(random[k]):
                fixed = new - ((new - old) @ E) @ E.T
                value = drift.cross_map(fixed, old, tasks)
                values.append(value)
                random_rows.append(restored_row(base, 'Random replicate', k, value,
                                                replicate=rep, k_effective=E.shape[1]))
            rows.append(restored_row(base, 'Random mean (10)', k, float(np.mean(values)),
                                     k_effective=k, mAP_restored_std=float(np.std(values)),
                                     restoration_ratio_std=float(np.std(values) / abs(base['forgetting_proxy']))))
            p.write_csv(str(root / 'A2_random_replicates.csv'), random_rows)
            p.write_csv(str(root / 'A2_restoration.csv'), rows)
            progress(root, 'A2_random', model=model, gallery_stage=j, k=k,
                     completed_rows=len(rows), random_replicates=len(random_rows))
    assert len(rows) == 54 and len(random_rows) == 90
    means = []
    names = ['Label-free identity stage1', 'Label-free identity stage3',
             'Label-free nuisance stage1', 'Identity oracle', 'FAP residual PCA', 'Random mean (10)']
    for name, k in itertools.product(names, KS):
        selected = [r for r in rows if r['basis'] == name and r['k'] == k]
        assert len(selected) == 3
        ratio = float(np.mean([r['restoration_ratio'] for r in selected]))
        oracle_ratio = float(np.mean([r['restoration_ratio'] for r in rows
                                      if r['basis'] == 'Identity oracle' and r['k'] == k]))
        means.append(dict(model=model, basis=name, k=k, old_domains=3,
                          mean_restoration_ratio=ratio, identity_oracle_ratio=oracle_ratio,
                          fraction_of_identity_oracle=ratio / oracle_ratio))
    p.write_csv(str(root / 'A2_summary.csv'), means)
    dump(root / 'complete.json', dict(status='complete', model=model, A1_property_rows=18,
         A1_dimension_combinations=27, A2_rows=54, random_replicate_rows=90,
         oracle_self_checks=9, oracle_self_check_passed=True))
    progress(root, 'complete', model=model, A2_rows=len(rows))


def aggregate(work):
    roots = [work / 'A' / m for m in ['F', 'HF']]
    for root in roots:
        assert json.loads((root / 'complete.json').read_text())['status'] == 'complete'
    tables = {}
    for name in ['A1_subspaces', 'A1_estimates', 'A1_stability', 'A2_restoration',
                 'A2_summary', 'A2_baselines', 'A2_random_replicates', 'oracle_self_check']:
        tables[name] = [r for root in roots for r in read_csv(root / (name + '.csv'))]
        p.write_csv(str(work / 'A' / (name + '.csv')), tables[name])
    means = {(r['model'], r['basis'], int(r['k'])): r for r in tables['A2_summary']}
    gates = []
    eligible = {}
    for model in ['F', 'HF']:
        eligible[model] = []
        for stage in [1, 3]:
            r = means[(model, 'Label-free identity stage%d' % stage, 16)]
            ratio, fraction = float(r['mean_restoration_ratio']), float(r['fraction_of_identity_oracle'])
            passed = ratio >= .4 and fraction >= .7
            if passed:
                eligible[model].append(stage)
            gates.append(dict(model=model, test='identity_k16', estimation_stage=stage,
                              mean_restoration_ratio=ratio, fraction_of_oracle=fraction,
                              passed=passed, criterion='mean >= 0.40 and oracle fraction >= 0.70'))
        r = means[(model, 'Label-free nuisance stage1', 2)]
        ratio = float(r['mean_restoration_ratio'])
        gates.append(dict(model=model, test='nuisance_k2', estimation_stage=1,
                          mean_restoration_ratio=ratio, passed=ratio <= .1,
                          criterion='mean <= 0.10'))
    oracle_ok = all(r['passed'] == 'True' and float(r['max_abs_delta']) < 1e-9
                    for r in tables['oracle_self_check'])
    nuisance_ok = all(r['passed'] for r in gates if r['test'] == 'nuisance_k2')
    passed = oracle_ok and all(eligible[m] for m in ['F', 'HF']) and nuisance_ok
    chosen = {}
    selected_k = None
    if passed:
        common = [s for s in [3, 1] if all(s in eligible[m] for m in ['F', 'HF'])]
        chosen = {m: common[0] if common else next(s for s in [3, 1] if s in eligible[m])
                  for m in ['F', 'HF']}
        selected_k = 8 if all(float(means[(m, 'Label-free identity stage%d' % chosen[m], 8)]
                                  ['fraction_of_identity_oracle']) >= .9 for m in ['F', 'HF']) else 16
    gate = dict(stage='A', passed=bool(passed), oracle_self_check_passed=oracle_ok,
                oracle_checks=len(tables['oracle_self_check']),
                oracle_max_abs_delta=max(float(r['max_abs_delta']) for r in tables['oracle_self_check']),
                eligible_identity_stages=eligible, selected_diagnostic_stage=chosen,
                selected_B_k=selected_k, identity_requirement_passed=all(bool(eligible[m]) for m in ['F', 'HF']),
                nuisance_requirement_passed=nuisance_ok,
                next_action='Proceed to implementation and U checks' if passed else 'STOP; publish Stage A only')
    dump(work / 'A/gate.json', gate)
    p.write_csv(str(work / 'A/gate_details.csv'), gates)
    dump(work / 'manifests/status.json', dict(phase='A_complete', status=
         'A_passed_ready_for_U' if passed else 'A_failed_stopped', gate=gate))

    explanations = []
    for model in ['F', 'HF']:
        identity_rows = [r for r in gates if r['model'] == model and r['test'] == 'identity_k16']
        explanations.append('%s k16 identity estimates (stage1/stage3): recovery %.3f%% / %.3f%%; '
                            'fraction of the identity oracle %.3f%% / %.3f%%. %s.' %
                            (model, *[100 * r['mean_restoration_ratio'] for r in identity_rows],
                             *[100 * r['fraction_of_oracle'] for r in identity_rows],
                             'PASS' if eligible[model] else 'FAIL: neither estimate meets both thresholds'))
    nuisance_rows = [r for r in gates if r['test'] == 'nuisance_k2']
    explanations.append('Nuisance k2 recovery: F %.3f%%, HF %.3f%%; %s the 10%% upper limit.' %
                        (100 * nuisance_rows[0]['mean_restoration_ratio'],
                         100 * nuisance_rows[1]['mean_restoration_ratio'],
                         'both pass' if nuisance_ok else 'at least one fails'))

    lines = ['# idAPR staged experiment', '', 'Run: `%s`' % work, '',
             '**Stage A: %s.** %s.' % ('PASS' if passed else 'FAIL', gate['next_action']), '',
             *[value + '\n' for value in explanations],
             'The fixed protocol uses k = 2, 8, 16, car/fit videos for label-free fitting, '
             'and exact S3 all-class retrieval for the oracle regression. All recovery ratios '
             'below are the equal mean of the three per-domain ratios, not a ratio of averaged mAP values.', '',
             '## Table A1 — Label-free subspace properties', '',
             'Angles are median/max in degrees against the common S3 F_stage1 fit oracle. '
             'Energy shares use held-out car videos and the P1 ratio-of-sums definition; '
             'real domain shifts use Clear as the reference. Stability compares same-dimensional '
             'stage1 and stage3 estimates. Stage4 uses FAP, photo and student views; the earlier '
             'stages use FAP only.', '',
             '| Model/stage | Space | k | Identity angle | Weather angle | FAP angle | Real shift energy | Identity energy | FAP energy | Stage1↔3 angle |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in tables['A1_subspaces']:
        def pair(prefix):
            return '%.2f/%.2f' % (float(r[prefix + '_median_deg']), float(r[prefix + '_max_deg']))
        lines.append('| %s/%s | %s | %s | %s | %s | %s | %.2f%% | %.2f%% | %.2f%% | %s |' %
                     (r['model'], r['stage'], r['kind'], r['k'], pair('identity_oracle'),
                      pair('weather_oracle'), pair('fap_residual_pca'),
                      100 * float(r['real_cross_domain_energy_share']),
                      100 * float(r['identity_energy_share']),
                      100 * float(r['fap_residual_energy_share']),
                      pair('stage1_stage3') if int(r['stage']) != 4 else '—'))
    lines += ['', 'All 54 (k_id, k_nuis) combinations and complete angle lists are retained in '
              '[A1_estimates.csv](A/A1_estimates.csv); distinct subspaces are in '
              '[A1_subspaces.csv](A/A1_subspaces.csv).', '',
              '## Table A2 — Restoration of old-domain forgetting', '',
              '| Model | Subspace | k | Mean recovery | Fraction of identity oracle |',
              '|---|---|---:|---:|---:|']
    for r in tables['A2_summary']:
        lines.append('| %s | %s | %s | %.3f%% | %.3f%% |' %
                     (r['model'], r['basis'], r['k'], 100 * float(r['mean_restoration_ratio']),
                      100 * float(r['fraction_of_identity_oracle'])))
    lines += ['', 'Identity oracle self-check: **%s**, %d/%d cells, maximum absolute discrepancy %.12g '
              '(required < 1e-9). The six original mAP/forgetting fields and query counts are checked '
              'for every cell.' % ('PASS' if oracle_ok else 'FAIL',
              sum(r['passed'] == 'True' for r in tables['oracle_self_check']),
              len(tables['oracle_self_check']), gate['oracle_max_abs_delta']), '',
              'Per-domain values: [A2_restoration.csv](A/A2_restoration.csv). '
              'All ten random replicates: [A2_random_replicates.csv](A/A2_random_replicates.csv). '
              'Oracle regression: [oracle_self_check.csv](A/oracle_self_check.csv).', '',
              '### Stage A gate', '', '| Model | Test | Estimate stage | Recovery | Oracle fraction | Pass |',
              '|---|---|---:|---:|---:|---|']
    for r in gates:
        lines.append('| %s | %s | %s | %.3f%% | %s | %s |' %
                     (r['model'], r['test'], r['estimation_stage'], 100 * r['mean_restoration_ratio'],
                      '%.3f%%' % (100 * r['fraction_of_oracle']) if 'fraction_of_oracle' in r else '—',
                      'PASS' if r['passed'] else 'FAIL'))
    lines += ['', 'Identity gate: each model needs at least one of stage1/stage3 k16 estimates '
              'to recover ≥40% and ≥70% of the same-k identity oracle. Nuisance k2 must recover '
              '≤10% in both models. Both estimates are retained regardless of the gate.', '',
              'Selected B dimension: %s.' % (selected_k if selected_k else 'not selected; A failed'), '',
              '## Table U — Unit and smoke checks', '',
              '| Check | Status | Reason |', '|---|---|---|']
    reason = 'Stage A did not pass; stopped as instructed' if not passed else 'Pending; Stage A has passed'
    for name in ['U1 CPU loss/gradient tests', 'U2 default-off bitwise parity', 'U3 B1/B2/B3 smoke training']:
        lines.append('| %s | NOT RUN | %s |' % (name, reason))
    lines += ['', '## Table B — Night training and forgetting', '',
              '| Group | Status | Reason |', '|---|---|---|']
    for name in ['B0 extra Overcast/Rainy evaluation', 'HF stage3 extra old-domain evaluation', 'B1', 'B2', 'B3']:
        lines.append('| %s | NOT RUN | %s |' % (name, reason))
    lines += ['', 'Stage B gate: not evaluated because Stage B has not run. No new training seeds '
              'were added.', '', '## Protocol clarifications and deviations', '',
              '- HF feature files are located under `/data1/liuyeqiang/damot_ics_followup_20261029/results/feats`, '
              'as recorded by the existing probe analysis; F uses the supplied RUN directory. '
              'All 40 input NPZ hashes and resolved paths are recorded in `manifests/feature_inputs.json`.',
              '- The label-free estimators use car fit RoIs only. The diagnostic oracle and A2 retrieval '
              'retain all S3 classes and the common F_stage1 reference so the requested oracle regression '
              'has exactly the historical protocol. Test energies use car only, as in P1.',
              '- Image IDs are namespaced by weather/video/frame only if the stored IDs collide between '
              'physical images. Each fit records whether this was necessary; no RoIs are removed.',
              '- Random controls use seed 0, ten draws per dimension, shared across F and HF; '
              'all replicates are reported. Stage A never loads a model on a GPU.',
              '', 'Source code, inputs, fixed gate thresholds and logs are included in the publication '
              'bundle. Weights and feature NPZ files remain on the server.', '']
    (work / 'SUMMARY.md').write_text('\n'.join(lines))
    print(json.dumps(gate), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', type=Path, required=True)
    ap.add_argument('--model', choices=['F', 'HF'])
    ap.add_argument('--aggregate', action='store_true')
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.work_dir)
    elif args.model:
        root = args.work_dir / 'A' / args.model
        try:
            run_model(args.work_dir, args.model)
        except Exception:
            dump(root / 'failure.json', dict(error=traceback.format_exc()))
            raise
    else:
        ap.error('Select --model or --aggregate')


if __name__ == '__main__':
    main()
