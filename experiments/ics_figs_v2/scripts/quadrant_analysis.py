"""Fixed-basis, full-test metrics and method-matched PCA/t-SNE displays."""
from common import *
import argparse
import time
import numpy as np
import evidence_cpu as evidence
import ics_probe_analyze as p
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUT = ROOT / 'quadrant'
SPACE_ORDER = ['INT', 'Random', 'ID', 'W', 'ICS', 'Complement', 'Full']
PLOT_SPACES = ['INT', 'Random', 'ID', 'Complement']
SPACE_TITLES = {'INT': 'INT 2D', 'Random': 'Random 2D (R_00)',
                'ID': 'ID oracle 2D', 'Complement': 'INT complement'}
WEATHER_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']


def report(model, phase):
    print('%s %s %s' % (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), model, phase), flush=True)
    write_json(OUT / ('progress_%s.json' % model), dict(model=model, phase=phase, time=time.time()))


def fixed_spaces(Z, model):
    folder = EVIDENCE / 'part1' / model / 'bases'
    bases = {}
    for name in ['INT', 'ID', 'W', 'ICS']:
        bases[name] = np.load(str(folder / ('%s_00.npy' % name)), allow_pickle=False)
    for j in range(10):
        bases['R_%02d' % j] = np.load(str(folder / ('R_%02d.npy' % j)), allow_pickle=False)
    for name, basis in bases.items():
        assert basis.shape == (256, 2), (name, basis.shape)
        assert np.isfinite(basis).all()
        assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-8), name
    for space in SPACE_ORDER:
        if space == 'Full':
            yield space, 0, Z
        elif space == 'Complement':
            E = bases['INT']
            with historical_probe_blas():
                X = Z - (Z @ E) @ E.T
            assert np.max(np.abs(X @ E)) < 1e-9
            yield space, 0, X
        elif space == 'Random':
            for j in range(10):
                with historical_probe_blas():
                    X = Z @ bases['R_%02d' % j]
                yield space, j, X
        else:
            with historical_probe_blas():
                X = Z @ bases[space]
            yield space, 0, X


def euclidean_map(X, tasks):
    # The same 4,000 predeclared queries and all their eligible galleries are used
    # in every space. No feature normalization is applied after projection.
    aps = []
    for query, gallery, positive in tasks:
        scores = -np.linalg.norm(X[gallery] - X[query], axis=1)
        aps.append(p.average_precision(scores, positive))
    return 100 * float(np.mean(aps))


def video_groups(video, instance):
    groups, silhouette_groups = [], []
    for v in np.unique(video):
        indices = np.flatnonzero(video == v)
        ids = instance[indices]
        unique, counts = np.unique(ids, return_counts=True)
        groups.append((v, indices, ids))
        keep_ids = unique[counts >= 3]
        if len(keep_ids) < 2:
            continue
        eligible = np.isin(ids, keep_ids)
        silhouette_groups.append((v, indices[eligible], ids[eligible]))
    return groups, silhouette_groups


def clustering_metrics(X, groups, silhouette_groups):
    sil_weighted, sil_rois = 0., 0
    for _, indices, ids in silhouette_groups:
        value = float(silhouette_score(X[indices], ids, metric='euclidean'))
        sil_weighted += len(indices) * value
        sil_rois += len(indices)
    within_weighted, var_rois, var_videos = 0., 0, 0
    for _, indices, ids in groups:
        xx = X[indices]
        total = float(np.sum((xx - xx.mean(0)) ** 2))
        if total <= 1e-20:
            continue
        within = 0.
        for identity in np.unique(ids):
            same = xx[ids == identity]
            within += float(np.sum((same - same.mean(0)) ** 2))
        ratio = within / total
        assert -1e-10 <= ratio <= 1 + 1e-10
        within_weighted += len(indices) * ratio
        var_rois += len(indices)
        var_videos += 1
    assert sil_rois and var_rois
    return dict(silhouette=sil_weighted / sil_rois,
                within_var_ratio=within_weighted / var_rois,
                n_videos_used=len(silhouette_groups), n_silhouette_rois=sil_rois,
                n_variance_videos=var_videos, n_variance_rois=var_rois)


def night_clear(X, data, fit):
    wf, wt = data['weather'][fit], data['weather'][~fit]
    use_f = np.isin(wf, ['night', evidence.DOMAINS[0]])
    use_t = np.isin(wt, ['night', evidence.DOMAINS[0]])
    return 100 * p.probe_predictions(X[fit][use_f], (wf[use_f] == 'night').astype(int),
                                     X[~fit][use_t], (wt[use_t] == 'night').astype(int), 0)[0]


def original_selfchecks(model, rows):
    old = {r['basis']: r for r in read_csv(EVIDENCE / 'part1' / model / 'deletion.csv')}
    expected_rounded = {
        'F': {'Full': [57.314, 34.513], 'Complement': [57.689, 32.782]},
        'HF': {'Full': [57.489, 36.358], 'Complement': [57.510, 34.257]},
    }
    checks = []
    for space, reference in [('Full', 'Full'), ('Complement', 'INT')]:
        row = next(r for r in rows if r['space'] == space)
        for i, (metric, old_metric) in enumerate([('id_mAP_cos', 'id_mAP_pp'), ('weather_bacc', 'weather_bacc_pp')]):
            actual, exact = row[metric], float(old[reference][old_metric])
            rounded = expected_rounded[model][space][i]
            check = dict(model=model, space=space, metric=metric, actual=actual, reference_exact=exact,
                         absolute_error=abs(actual - exact), reference_prompt=rounded,
                         pass_check=bool(abs(actual - exact) < 1e-7 and abs(actual - rounded) < .001))
            checks.append(check)
            assert check['pass_check'], check
    write_csv(OUT / ('selfchecks_%s.csv' % model), checks)
    print('QUADRANT_SELFCHECK_PASS', model, len(checks), flush=True)
    return checks


def decisions(model, rows):
    one = {r['space']: r for r in rows if r['space'] != 'Random'}
    random = [r for r in rows if r['space'] == 'Random']
    assert len(random) == 10
    weather = np.array([r['weather_bacc'] for r in random])
    threshold = float(weather.mean() + 3 * weather.std(ddof=0))
    comparisons = {}
    for metric in ['id_mAP_euc', 'silhouette']:
        values = np.array([r[metric] for r in random])
        lo, hi = float(values.mean() - 2 * values.std(ddof=0)), float(values.mean() + 2 * values.std(ddof=0))
        observed = one['INT'][metric]
        comparisons[metric] = dict(INT=observed, random_mean=float(values.mean()),
                                   random_sd=float(values.std(ddof=0)), band_low=lo, band_high=hi,
                                   result='higher' if observed > hi else 'lower' if observed < lo else 'comparable')
    ratio = one['Complement']['id_mAP_cos'] / one['Full']['id_mAP_cos']
    return dict(model=model,
        P1=dict(status='PASS' if one['INT']['weather_bacc'] > threshold else 'FAIL',
                INT_weather_bacc=one['INT']['weather_bacc'], random_mean=float(weather.mean()),
                random_sd=float(weather.std(ddof=0)), threshold=threshold),
        P2=dict(status='PASS' if all(one['INT'][m] < one['ID'][m] for m in ['id_mAP_euc', 'silhouette']) else 'FAIL',
                INT_id_mAP=one['INT']['id_mAP_euc'], ID_id_mAP=one['ID']['id_mAP_euc'],
                INT_silhouette=one['INT']['silhouette'], ID_silhouette=one['ID']['silhouette']),
        P3=comparisons,
        P4=dict(status='PASS' if ratio >= .98 else 'FAIL', retention_ratio=ratio,
                complement_cosine_mAP=one['Complement']['id_mAP_cos'], full_cosine_mAP=one['Full']['id_mAP_cos']))


def source_files(model):
    paths = [SCRIPTS / name for name in ['evidence_cpu.py', 'ics_probe_analyze.py', 'idapr_basis.py']]
    folder = EVIDENCE / 'part1' / model
    paths += [folder / name for name in ['protocol.json', 'figure_protocol.json', 'deletion.csv']]
    paths += sorted((folder / 'bases').glob('*.npy'))
    paths += [evidence.INPUTS[model] / (domain + '.npz') for domain in evidence.DOMAINS]
    return [source_record(path) for path in paths]


def metrics(model, data, Z, fit, sources):
    video, frame, inst = [data[k][~fit] for k in ['video', 'frame_id', 'instance_id']]
    tasks = p.build_retrieval(video, frame, inst, 4, 30, 4000, 0)
    assert len(tasks) == 4000
    groups, sil_groups = video_groups(video, inst)
    write_json(OUT / ('eligible_videos_%s.json' % model), dict(
        silhouette=[dict(video=str(v), rois=len(idx), identities=len(np.unique(ids))) for v, idx, ids in sil_groups],
        variance=[dict(video=str(v), rois=len(idx), identities=len(np.unique(ids))) for v, idx, ids in groups]))
    digest = hashlib.sha256()
    for q, g, pos in tasks:
        digest.update(np.array([q, len(g)], dtype=np.int64).tobytes())
        digest.update(np.asarray(g, dtype=np.int64).tobytes())
        digest.update(np.asarray(pos, dtype=np.uint8).tobytes())
    rows = []
    for space, replicate, X in fixed_spaces(Z, model):
        report(model, 'metrics_%s_%02d' % (space, replicate))
        xt = X[~fit]
        row = dict(model=model, space=space, replicate=replicate,
                   id_mAP_euc=euclidean_map(xt, tasks),
                   id_mAP_cos=(100 * evidence.cross_map(xt, xt, tasks) if space in ['Full', 'Complement'] else ''))
        clustering = clustering_metrics(xt, groups, sil_groups)
        with historical_probe_blas():
            weather = 100 * p.weather_probe(X[fit], data['weather'][fit], xt, data['weather'][~fit], 0)
            binary = night_clear(X, data, fit)
        row.update(silhouette=clustering['silhouette'], within_var_ratio=clustering['within_var_ratio'],
                   weather_bacc=weather, nightclear_bacc=binary, n_rois=len(xt),
                   n_videos_used=clustering['n_videos_used'],
                   n_silhouette_rois=clustering['n_silhouette_rois'],
                   n_variance_videos=clustering['n_variance_videos'], n_variance_rois=clustering['n_variance_rois'],
                   n_test_videos=len(groups), n_retrieval_queries=len(tasks))
        rows.append(row)
        write_csv(OUT / ('quadrant_metrics_%s.csv' % model), rows)
        print('METRIC', json.dumps(row, allow_nan=False), flush=True)
    checks = original_selfchecks(model, rows)
    gate = decisions(model, rows)
    write_json(OUT / ('decisions_%s.json' % model), gate)
    verify_sources(sources)
    write_json(OUT / ('metrics_protocol_%s.json' % model), dict(status='PASS', source_files=sources,
        load_function='Unmodified evidence_cpu.load(model); stage4 clean features L2 normalized once',
        fit_rois=int(fit.sum()), test_rois=int((~fit).sum()), fit_videos=len(np.unique(data['video'][fit])),
        test_videos=len(np.unique(video)), dimensions=256, category='car', seed=0,
        retrieval=dict(gap=4, window=30, max_queries=4000, queries=len(tasks), task_sha256=digest.hexdigest(),
                       scoring='Negative Euclidean distance in every space; additional historical cosine for Full/Complement'),
        silhouette='Exact sklearn silhouette_score, Euclidean, within video; identity count>=3 and >=2 eligible identities; RoI-weighted video mean',
        within_var_ratio='Within-identity SSE / video-total SSE in every nonconstant test video, all identities; RoI-weighted video mean',
        random_sd_ddof=0, basis_reestimated=False, source_mutation=False, original_selfchecks=checks,
        numerical_compatibility='Shell thread variables are 4; load/projections/logistic probes replay historical MKL=2; all other operations restore MKL=4',
        compatibility_evidence='../ics_evidence/logs/part1_F_terminal_transcript.txt and logs/probe_numerical_diagnostic.csv'))
    report(model, 'metrics_complete')
    return rows


def figures(model, data, Z, fit, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    source = EVIDENCE / 'part1' / model / 'figure_protocol.json'
    protocol = json.loads(source.read_text())
    sample = np.asarray(protocol['quadrant_indices'], dtype=int)
    highlights = protocol['highlighted_identities']
    assert len(sample) == 4000 and len(np.unique(sample)) == len(sample)
    assert np.all(~fit[sample]) and len(highlights) == 10
    identities = np.array(['%s|%d' % (v, i) for v, i in zip(data['video'], data['instance_id'])])
    assert all(np.any(identities[sample] == h) for h in highlights)
    selected = {(space, replicate): X[sample].copy() for space, replicate, X in fixed_spaces(Z, model)
                if space in PLOT_SPACES and replicate == 0}
    row_by = {r['space']: r for r in rows if int(r['replicate']) == 0}
    coordinates = []
    embeddings = {}
    for method in ['pca', 'tsne']:
        for space in PLOT_SPACES:
            report(model, '%s_%s' % (method, space))
            X = selected[(space, 0)]
            if method == 'pca':
                coords = PCA(n_components=2, random_state=0).fit_transform(X)
            else:
                xx = PCA(n_components=50, random_state=0).fit_transform(X) if X.shape[1] > 50 else X
                coords = TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(xx)
            assert coords.shape == (4000, 2) and np.isfinite(coords).all()
            embeddings[(method, space)] = coords
            for original, coord in zip(sample, coords):
                identity = identities[original]
                coordinates.append(dict(model=model, method=method, space=space, data_index=int(original),
                                        x=float(coord[0]), y=float(coord[1]), weather=str(data['weather'][original]),
                                        video=str(data['video'][original]), instance_id=int(data['instance_id'][original]),
                                        highlighted_identity_index=highlights.index(identity) if identity in highlights else -1))
        fig, axes = plt.subplots(2, 4, figsize=(7., 4.9))
        for col, space in enumerate(PLOT_SPACES):
            coords = embeddings[(method, space)]
            row = row_by[space]
            for domain, color in zip(evidence.DOMAINS, WEATHER_COLORS):
                keep = data['weather'][sample] == domain
                axes[0, col].scatter(coords[keep, 0], coords[keep, 1], s=2, alpha=.45,
                                     color=color, linewidths=0, rasterized=True)
            axes[1, col].scatter(coords[:, 0], coords[:, 1], s=1.7, color='.86', linewidths=0, rasterized=True)
            for j, identity in enumerate(highlights):
                keep = identities[sample] == identity
                axes[1, col].scatter(coords[keep, 0], coords[keep, 1], s=8,
                                     color=plt.get_cmap('tab20')(2*j), linewidths=0, rasterized=True)
            for ax in axes[:, col]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(.025, .975, 'ID mAP %.1f\nSil %.2f\nWeather bAcc %.1f' %
                        (float(row['id_mAP_euc']), float(row['silhouette']), float(row['weather_bacc'])),
                        transform=ax.transAxes, va='top', ha='left', fontsize=9,
                        bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1.2))
            axes[0, col].set_title(SPACE_TITLES[space], fontsize=9)
        axes[0, 0].set_ylabel('Weather colors', fontsize=9)
        axes[1, 0].set_ylabel('Ten fixed identities', fontsize=9)
        handles = [Line2D([0], [0], marker='o', ls='', color=c, label=evidence.DISPLAY[d], markersize=4)
                   for d, c in zip(evidence.DOMAINS, WEATHER_COLORS)]
        fig.suptitle('%s: %s; same 4,000 test RoIs' % (model, method.upper() if method == 'pca' else 't-SNE'),
                     fontsize=11, y=.995)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.51, .965), ncol=5,
                   frameon=False, fontsize=9, columnspacing=1., handletextpad=.3)
        fig.text(.5, .015, 'mAP / bAcc: %; all-test metrics before visualization. Random panel: fixed R_00.',
                 ha='center', fontsize=9)
        fig.subplots_adjust(left=.065, right=.99, top=.83, bottom=.07, hspace=.10, wspace=.06)
        stem = OUT / ('quadrant_%s_%s' % (method, model))
        fig.savefig(str(stem) + '.pdf', dpi=300)
        fig.savefig(str(stem) + '.png', dpi=300)
        plt.close(fig)
    write_csv(OUT / ('plot_coordinates_%s.csv' % model), coordinates)
    write_json(OUT / ('figure_protocol_%s.json' % model), dict(
        source_protocol=source_record(source), quadrant_indices=sample.tolist(),
        highlighted_identities=highlights, random_plot_replicate=0, coordinate_fit='fixed 4,000 displayed test RoIs',
        metrics_fit='all test RoIs, retrieval task cap 4,000 as specified; not the display subset',
        pca='PCA(n_components=2, random_state=0) independently in every space',
        tsne='2D inputs unchanged; complement PCA(n_components=50, random_state=0); TSNE(n_components=2, perplexity=30, init=pca, random_state=0)',
        annotation_retrieval='Negative Euclidean mAP, measured before PCA/t-SNE', min_font_pt=9,
        weather_colors=WEATHER_COLORS, points_reselected=False, bases_reestimated=False))
    report(model, 'figures_complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['F', 'HF'], required=True)
    parser.add_argument('--phase', choices=['metrics', 'figures', 'all'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report(args.model, 'load')
    sources = source_files(args.model)
    with historical_probe_blas():
        data, bank, Z, fit = evidence.load(args.model)
    assert Z.shape[1] == 256 and np.all(data['category'] == 1)
    old = json.loads((EVIDENCE / 'part1' / args.model / 'protocol.json').read_text())
    assert int(fit.sum()) == old['fit_rois'] and int((~fit).sum()) == old['test_rois']
    assert sorted(np.unique(data['video'][~fit]).tolist()) == sorted(old['test_videos'])
    rows = metrics(args.model, data, Z, fit, sources) if args.phase != 'figures' else read_csv(OUT / ('quadrant_metrics_%s.csv' % args.model))
    if args.phase != 'metrics':
        figures(args.model, data, Z, fit, rows)
        verify_sources(sources)
    write_json(OUT / ('complete_%s.json' % args.model), dict(status='complete', phase=args.phase,
               source_files_unchanged=True, model=args.model, metric_rows=len(rows), cpu_only=True))


if __name__ == '__main__':
    main()
