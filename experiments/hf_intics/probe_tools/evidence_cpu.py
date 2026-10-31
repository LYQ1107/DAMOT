"""Prespecified ICS evidence, BDD car / video split seed 0; CPU only."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import ics_probe_analyze as p
from idapr_basis import label_free_bases

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ['clear_holdout_45v', 'overcast', 'rainy', 'dawn_dusk', 'night']
INPUTS = {
    'F': Path('/data1/liuyeqiang/damot_rebuttal_20261026/results/ics_probe/feats/F_stage4_e5'),
    'HF': Path('/data1/liuyeqiang/damot_ics_followup_20261029/results/feats/HF_stage4_e5')}
DISPLAY = dict(zip(DOMAINS, ['Clear', 'Overcast', 'Rainy', 'Dawn/Dusk', 'Night']))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csvout(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def progress(root, phase):
    dump(root / 'progress.json', dict(phase=phase, time=time.time()))
    print(phase, flush=True)


def sign_basis(E):
    E = E.copy()
    for j in range(E.shape[1]):
        if E[np.argmax(np.abs(E[:, j])), j] < 0:
            E[:, j] *= -1
    return E


def load(model):
    data, bank = p.load_label(str(INPUTS[model].parent), INPUTS[model].name, DOMAINS, [1])
    Z = p.normalize(data['feat_clean'].astype(float))
    assert np.isfinite(Z).all()
    fit = p.split_mask(data['video'], 0)
    assert not set(data['video'][fit]) & set(data['video'][~fit])
    return data, bank, Z, fit


def bases(data, bank, Z, fit):
    frames = np.array(['%s|%s|%d' % (w, v, f) for w, v, f in
                       zip(data['weather'], data['video'], data['frame_id'])])
    views = [p.normalize(data['feat_' + name].astype(float)) for name in ['fap', 'photo', 'student']]
    _, INT, stats = label_free_bases(Z[fit], frames[fit], [v[fit] for v in views], 2, 2)
    residual = np.concatenate([v[fit] - Z[fit] for v in views])
    direct = p.top_eig(residual.T @ residual / len(residual), 2)
    assert np.max(p.principal_angles_deg(INT, direct)) < 1e-5
    W = p.between_weather_rows(Z[fit], data['weather'][fit], data['category'][fit])
    ID = p.identity_rows(Z[fit], data['video'][fit], data['instance_id'][fit], data['category'][fit])
    rng = np.random.RandomState(0)
    B = dict(INT=[INT], ICS=[p.orth(bank['pcs'].T.astype(float))],
             W=[p.top_eig(W.T @ W, 2)], ID=[p.top_eig(ID.T @ ID, 2)],
             R=[p.orth(rng.randn(Z.shape[1], 2)) for _ in range(10)])
    B = {k: [sign_basis(e) for e in bb] for k, bb in B.items()}
    for bb in B.values():
        for e in bb:
            assert np.allclose(e.T @ e, np.eye(2), atol=1e-8)
    return B, stats, frames


def cross_map(query, gallery, tasks):
    q, g = p.normalize(query), p.normalize(gallery)
    return float(np.mean([p.average_precision(g[gg] @ q[qq], pos) for qq, gg, pos in tasks]))


def icc(values, groups, domains):
    """RoI-weighted variance of fitted group means / variance of RoI values."""
    total = float(np.var(values))
    def component(labels):
        _, inverse = np.unique(labels, return_inverse=True)
        counts = np.bincount(inverse)
        means = np.bincount(inverse, weights=values) / counts
        return float(np.sum(counts * (means - values.mean()) ** 2) / len(values))
    return dict(ICC=component(groups) / total, domain_share=component(domains) / total,
                total_variance=total, frame_between_variance=component(groups),
                domain_between_variance=component(domains), rois=len(values),
                frames=len(np.unique(groups)))


def measure(raw, data, fit, tasks):
    wf, wt = data['weather'][fit], data['weather'][~fit]
    five = p.weather_probe(raw[fit], wf, raw[~fit], wt, 0)
    fbin, tbin = np.isin(wf, ['night', DOMAINS[0]]), np.isin(wt, ['night', DOMAINS[0]])
    two = p.probe_predictions(raw[fit][fbin], (wf[fbin] == 'night').astype(int),
                              raw[~fit][tbin], (wt[tbin] == 'night').astype(int), 0)[0]
    return dict(weather_bacc_pp=100 * five, night_clear_bacc_pp=100 * two,
                fisher=p.fisher_ratio(raw[~fit], wt, data['category'][~fit]),
                id_mAP_pp=100 * cross_map(raw[~fit], raw[~fit], tasks))


def resolve_image(path):
    pth = Path(path)
    if pth.is_file():
        return pth
    # Existing private read-only BDD view; no mount or link is modified.
    marker = '/data/views/'
    if marker in str(path):
        tail = str(path).split(marker, 1)[1]
        q = Path('/data1/liuyeqiang/ics_v2_20261030_184026/data_views') / tail
        if q.is_file():
            return q
    raise FileNotFoundError(str(path))


def plots(model, root, data, Z, fit, B, frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from PIL import Image
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fdir = root / 'figures'
    fdir.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(str(fdir / (name + '.pdf')), bbox_inches='tight')
        fig.savefig(str(fdir / (name + '.png')), dpi=200, bbox_inches='tight')
        plt.close(fig)
    te = ~fit
    unique, inv, count = np.unique(frames, return_inverse=True, return_counts=True)
    eligible = te & (count[inv] >= 3)
    rng = np.random.RandomState(0)
    chosen, xx, ticks = [], 0, []
    for d in DOMAINS:
        available = np.unique(frames[eligible & (data['weather'] == d)])
        n = min(8, len(available))
        assert n == 8, ('fewer than eight eligible frames', d, n)
        for frame in sorted(rng.choice(available, 8, replace=False)):
            chosen.append((frame, d, xx))
            xx += 1
        ticks.append((xx - 4.5, DISPLAY[d]))
        xx += 2
    csvout(root / 'plot_B_frames.csv', [dict(frame=f, weather=d, x=x) for f, d, x in chosen])
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, sharey=True)
    for ax, key in zip(axs, ['INT', 'ID']):
        for frame, d, x in chosen:
            m = frames == frame
            ax.scatter(np.full(m.sum(), x), Z[m] @ B[key][0][:, 0], s=13,
                       color=plt.get_cmap('tab10')(DOMAINS.index(d)), alpha=.7)
        ax.set_ylabel(key + ' $e_1$ projection')
        ax.grid(axis='y', alpha=.2)
    axs[-1].set_xticks([x for x, _ in ticks])
    axs[-1].set_xticklabels([d for _, d in ticks])
    fig.suptitle(model + ': shared frame shifts (fixed seed 0)')
    fig.tight_layout()
    save(fig, 'B_shared_shifts')

    inds = np.where(te)[0]
    E = B['INT'][0]
    projection = Z @ E[:, 0]
    order = inds[np.argsort(projection[inds])]
    picks, seen = [], set()
    for side, seq in [('low', order), ('high', order[::-1])]:
        selected = []
        for i in seq:
            identity = (str(data['video'][i]), int(data['instance_id'][i]))
            if identity not in seen:
                selected.append(int(i)); seen.add(identity)
            if len(selected) == 10:
                break
        picks.append((side, selected))
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.8))
    crop_rows = []
    for r, (side, selected) in enumerate(picks):
        for c, i in enumerate(selected):
            path = resolve_image(data['img_path'][i])
            image = Image.open(str(path)).convert('RGB')
            bbox = tuple(map(float, data['bbox'][i]))
            axes[r, c].imshow(image.crop(bbox))
            axes[r, c].set_title('%s\n%+.3f' % (DISPLAY[str(data['weather'][i])], projection[i]), fontsize=8)
            axes[r, c].axis('off')
            crop_rows.append(dict(side=side, column=c, index=i, weather=str(data['weather'][i]),
                                  video=str(data['video'][i]), instance_id=int(data['instance_id'][i]),
                                  frame_id=int(data['frame_id'][i]), e1=float(projection[i]),
                                  path=str(path), bbox=json.dumps(bbox)))
    fig.suptitle(model + ': INT $e_1$ extrema; ten unique tracks per end')
    fig.tight_layout()
    save(fig, 'C_INT_crops')
    csvout(root / 'plot_C_crops.csv', crop_rows)

    dose_keys = [k for k in data if k.startswith(('feat_dark_', 'feat_fog_', 'feat_fapk_'))]
    dose_mask = te & np.logical_or.reduce([np.isfinite(data[k]).all(1) for k in dose_keys])
    reference = p.identity_pairs(Z[dose_mask], frames[dose_mask], data['instance_id'][dose_mask], seed=0) @ E[:, 0]
    dose_rows = []
    for k in sorted(dose_keys):
        family, level = k[5:].split('_')
        level = float(level.replace('p', '.'))
        m = te & np.isfinite(data[k]).all(1)
        delta = (p.normalize(data[k][m].astype(float)) - Z[m]) @ E[:, 0]
        dose_rows.append(dict(family=family, level=level, strength=level if family == 'fog' else 1-level,
                              rois=int(m.sum()), mean=float(delta.mean()), std=float(delta.std()),
                              abs_mean=float(np.abs(delta).mean()), abs_std=float(np.abs(delta).std()),
                              identity_reference_mean=float(reference.mean()),
                              identity_reference_abs_mean=float(np.abs(reference).mean()),
                              identity_reference_pairs=len(reference)))
    csvout(root / 'dose_response.csv', dose_rows)
    for absolute in [False, True]:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
        for ax, family in zip(axes, ['dark', 'fog', 'fapk']):
            rows = sorted([r for r in dose_rows if r['family'] == family], key=lambda r: r['strength'])
            mean, std = ('abs_mean', 'abs_std') if absolute else ('mean', 'std')
            x = [0.] + [r['strength'] for r in rows]
            y = [0.] + [r[mean] for r in rows]
            sd = [0.] + [r[std] for r in rows]
            ax.errorbar(x, y, yerr=sd, fmt='o-', capsize=3, label='intervention')
            value = np.abs(reference).mean() if absolute else reference.mean()
            ax.axhline(value, ls='--', color='.4', label='other car, same frame')
            ax.set_title(family); ax.set_xlabel('Strength'); ax.grid(alpha=.2)
        axes[0].set_ylabel('Absolute $e_1$ change' if absolute else 'Signed $e_1$ change')
        axes[-1].legend(fontsize=7)
        fig.suptitle(model + ': mean ± RoI SD (not confidence intervals)')
        fig.tight_layout()
        save(fig, 'D_dose_response' + ('_absolute' if absolute else ''))

    rng = np.random.RandomState(0)
    sample = rng.choice(inds, min(4000, len(inds)), replace=False)
    A = Z[sample] @ E
    C = p.complement(Z[sample], E)
    reduced = PCA(n_components=50, random_state=0).fit_transform(C)
    T = TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(reduced)
    identities = np.array(['%s|%d' % (v, i) for v, i in zip(data['video'], data['instance_id'])])
    highlights = []
    for d in DOMAINS:
        ids, cnt = np.unique(identities[sample][data['weather'][sample] == d], return_counts=True)
        highlights.extend(ids[np.argsort(-cnt)[:2]])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
    for col, (coords, title) in enumerate([(A, 'INT plane'), (T, 'Complement (t-SNE)')]):
        for d in DOMAINS:
            m = data['weather'][sample] == d
            axes[0, col].scatter(coords[m, 0], coords[m, 1], s=4, alpha=.5, label=DISPLAY[d])
        axes[0, col].set_title(title + ': weather')
        axes[1, col].scatter(coords[:, 0], coords[:, 1], s=2, color='.85')
        for j, identity in enumerate(highlights):
            m = identities[sample] == identity
            axes[1, col].scatter(coords[m, 0], coords[m, 1], s=9, color=plt.get_cmap('tab20')(j*2))
        axes[1, col].set_title(title + ': ten highlighted identities')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, markerscale=3)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, 'INT_quadrant')
    dump(root / 'figure_protocol.json', dict(seed=0, common_y_B=True, crop_unique_globally=True,
         dose_signed_and_absolute=True, dose_error_bars='RoI SD, not CI', quadrant_indices=sample.tolist(),
         highlighted_identities=highlights))


def run(model):
    root = ROOT / 'part1' / model
    root.mkdir(parents=True, exist_ok=True)
    progress(root, 'load')
    data, bank, Z, fit = load(model)
    B, stats, frames = bases(data, bank, Z, fit)
    (root / 'bases').mkdir(exist_ok=True)
    for name, bb in B.items():
        for rep, E in enumerate(bb):
            np.save(str(root / 'bases' / ('%s_%02d.npy' % (name, rep))), E)
    dump(root / 'protocol.json', dict(model=model, fit_videos=np.unique(data['video'][fit]).tolist(),
         test_videos=np.unique(data['video'][~fit]).tolist(), seed=0, fit_rois=int(fit.sum()),
         test_rois=int((~fit).sum()), classes=['car'], feature_dimension=Z.shape[1],
         INT='uncentered covariance of normalized paired view minus normalized clean; fap/photo/student',
         e1_sign='largest-magnitude coordinate positive; no outcome-based sign choice',
         ICC='RoI-weighted variance of frame means / total variance; minimum three distinct cars',
         deletion='Z - Z E E.T; no new L2 row normalization before classifiers or Fisher',
         retrieval='cosine scoring as historical P2; gap4 window30 forward (f+4,f+34]; max4000 seed0',
         classifier='LogisticRegression C1 lbfgs balanced max_iter2000; fit standardization only',
         label_free_fit_stats=stats,
         inputs=[dict(path=str(INPUTS[model] / (d+'.npz')), sha256=sha(INPUTS[model] / (d+'.npz'))) for d in DOMAINS]))
    tasks = p.build_retrieval(data['video'][~fit], data['frame_id'][~fit], data['instance_id'][~fit], 4, 30, 4000, 0)
    baseline_map = cross_map(Z[~fit], Z[~fit], tasks)
    cross_rows = []
    for v in ['clean', 'fap', 'photo', 'student']:
        value = cross_map(Z[~fit], data['feat_'+v][~fit].astype(float), tasks)
        cross_rows.append(dict(model=model, view=v, queries=len(tasks), mAP_pp=100*value,
                               clean_mAP_pp=100*baseline_map, ratio=value/baseline_map, pass_095=value/baseline_map>=.95))
    csvout(root / 'cross_view_retrieval.csv', cross_rows)
    progress(root, 'ICC')
    # Each extracted GT instance occurs once per frame; audit distinct count.
    _, inv, cnt = np.unique(frames, return_inverse=True, return_counts=True)
    eligible = (~fit) & (cnt[inv] >= 3)
    for frame in np.unique(frames[eligible]):
        assert len(np.unique(data['instance_id'][frames == frame])) == (frames == frame).sum()
    icc_rows = []
    for name, bb in B.items():
        for rep, E in enumerate(bb):
            for coord in ['e1', 'norm2']:
                values = Z[eligible] @ E
                values = values[:, 0] if coord == 'e1' else np.linalg.norm(values, axis=1)
                icc_rows.append(dict(model=model, basis=name, replicate=rep, coordinate=coord,
                                    **icc(values, frames[eligible], data['weather'][eligible])))
    csvout(root / 'ICC_replicates.csv', icc_rows)
    progress(root, 'deletion_baseline')
    baseline = measure(Z, data, fit, tasks)
    assert abs(baseline['id_mAP_pp'] - 100*baseline_map) < 1e-9
    deletion = [dict(model=model, basis='Full', replicate=0, **baseline,
                     **{k+'_delta':0. for k in baseline}, fisher_drop_pct=0.)]
    for name, bb in B.items():
        for rep, E in enumerate(bb):
            progress(root, 'delete_%s_%d' % (name, rep))
            raw = Z - (Z @ E) @ E.T
            assert np.max(np.abs(raw @ E)) < 1e-10
            values = measure(raw, data, fit, tasks)
            deletion.append(dict(model=model, basis=name, replicate=rep, **values,
                                  **{k+'_delta':values[k]-baseline[k] for k in values},
                                  fisher_drop_pct=100*(1-values['fisher']/baseline['fisher'])))
            csvout(root / 'deletion_replicates.csv', deletion)
    icc_summary, deletion_summary = [], []
    for name in B:
        for coord in ['e1', 'norm2']:
            group = [r for r in icc_rows if r['basis']==name and r['coordinate']==coord]
            icc_summary.append(dict(model=model, basis=name, coordinate=coord, repeats=len(group),
                **{k:float(np.mean([r[k] for r in group])) for k in ['ICC','domain_share']},
                **{k+'_std':float(np.std([r[k] for r in group])) for k in ['ICC','domain_share']}))
    for name in ['Full'] + list(B):
        group = [r for r in deletion if r['basis']==name]
        keys = list(baseline) + [k+'_delta' for k in baseline] + ['fisher_drop_pct']
        deletion_summary.append(dict(model=model, basis=name, repeats=len(group),
             **{k:float(np.mean([r[k] for r in group])) for k in keys},
             **{k+'_std':float(np.std([r[k] for r in group])) for k in keys}))
    csvout(root / 'ICC.csv', icc_summary)
    csvout(root / 'deletion.csv', deletion_summary)
    gap = {r['basis']:r['ICC'] for r in icc_summary if r['coordinate']=='e1'}
    dint = next(r for r in deletion_summary if r['basis']=='INT')
    dump(root / 'gates.json', dict(cross_view_all_pass=all(r['pass_095'] for r in cross_rows),
         ICC_INT_minus_ID=gap['INT']-gap['ID'], ICC_pass=gap['INT']-gap['ID']>=.3,
         INT_fisher_drop_pct=dint['fisher_drop_pct'], INT_id_delta_pp=dint['id_mAP_pp_delta'],
         deletion_pass=dint['fisher_drop_pct']>=30 and abs(dint['id_mAP_pp_delta'])<=1))
    progress(root, 'plots')
    plots(model, root, data, Z, fit, B, frames)
    dump(root / 'complete.json', dict(status='complete', model=model, time=time.time()))
    progress(root, 'complete')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['F','HF'], required=True)
    run(ap.parse_args().model)
