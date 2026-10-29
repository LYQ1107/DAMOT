#!/usr/bin/env python
"""Offline analysis for the ICS subspace probe (CPU only, no mmcv/torch).

Question: is the subspace that ICS shrinks dominated by weather, and does it
carry little identity information?

Reads the npz files written by ics_probe_extract.py and produces
    <out>/csv/*.csv                 every number
    <out>/tables/*.tex              LaTeX tables (paper / supplementary)
    <out>/figs/*.pdf|png            figures
    <out>/summary.json

Experiments
    P1  energy share   : how much weather / FAP / identity variation falls in E
    P2  double dissoc. : weather probe and identity retrieval on E vs. E-perp
    P3  intervention   : shrink E (beta grid) vs. random / identity subspaces
    P4  dose response  : graded darkening / fog / FAP strength (Fig. 4)
    P5  stage evolution: ICS basis after each stage vs. each domain shift
    P6  paired weather : (VKITTI2 only) same object, same frame, other weather
Python 3.7 compatible (server: numpy 1.21, scipy 1.4, sklearn 0.22, mpl 3.5).
"""
import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import sys

import numpy as np

EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feat-root', required=True,
                   help='the --out-dir given to ics_probe_extract.py')
    p.add_argument('--label', required=True, help='main checkpoint label')
    p.add_argument('--domains', nargs='+', required=True,
                   help='domain names in display order (weather labels)')
    p.add_argument('--ref-domain', default=None,
                   help='reference weather for shift vectors (e.g. clear_holdout_45v)')
    p.add_argument('--stage-labels', nargs='*', default=[],
                   help='checkpoint label after each stage, in order (P5)')
    p.add_argument('--stage-domains', nargs='*', default=[],
                   help='domain adapted at each stage, same order (P5)')
    p.add_argument('--compare-labels', nargs='*', default=[],
                   help='other checkpoints whose ICS bank is compared on the '
                        'main features (e.g. a pairwise_residual run)')
    p.add_argument('--classes', default='1', help='category ids, e.g. "1" = car')
    p.add_argument('--m', type=int, default=0,
                   help='subspace dim; 0 = use the bank size of the checkpoint')
    p.add_argument('--betas', default='0,0.1,0.25,0.5,0.75,1.0')
    p.add_argument('--residual-view', default='fap',
                   help='view whose paired residual defines the offline FAP-PCA')
    p.add_argument('--paired', action='store_true',
                   help='VKITTI2 mode: pair RoIs by (scene, frame, instance) '
                        'across variants')
    p.add_argument('--ref-variant', default='clone')
    p.add_argument('--gap', type=int, default=4, help='min frame gap query->gallery')
    p.add_argument('--window', type=int, default=30, help='gallery frame window')
    p.add_argument('--max-queries', type=int, default=4000)
    p.add_argument('--n-random', type=int, default=50)
    p.add_argument('--montage', action='store_true',
                   help='read image crops for Fig. 3 (needs image paths)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def load_label(root, label, domains, classes):
    parts = []
    bank = None
    for d in domains:
        f = osp.join(root, label, d + '.npz')
        if not osp.exists(f):
            print('[warn] missing %s' % f)
            continue
        z = np.load(f, allow_pickle=False)
        keep = np.isin(z['category'], classes)
        part = {k: z[k][keep] for k in z.files
                if not k.startswith('bank_') and z[k].ndim >= 1 and
                z[k].shape[0] == z['category'].shape[0]}
        parts.append(part)
        if bank is None and 'bank_pcs' in z.files:
            bank = {k[5:]: z[k] for k in z.files if k.startswith('bank_')}
    if not parts:
        raise SystemExit('no features for label %s' % label)
    keys = set(parts[0])
    for p in parts[1:]:
        keys &= set(p)
    data = {k: np.concatenate([p[k] for p in parts], 0) for k in keys}
    return data, bank


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def orth(a):
    q, _ = np.linalg.qr(a)
    return q


def split_mask(groups, seed=0):
    """Deterministic 50/50 split on group names (video or scene)."""
    u = np.unique(groups)
    fit = set(g for g in u if int(hashlib.md5(('%s/%d' % (g, seed)).encode())
                                  .hexdigest()[:8], 16) % 2 == 0)
    if len(fit) in (0, len(u)):           # degenerate: alternate
        fit = set(u[::2])
    return np.array([g in fit for g in groups])


# --------------------------------------------------------------------------- #
# subspaces
# --------------------------------------------------------------------------- #
def top_eig(cov, m):
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    return v[:, np.argsort(w)[::-1][:m]]


def cov_of(rows, center=True):
    x = rows - rows.mean(0, keepdims=True) if center else rows
    return x.T @ x / max(len(x), 1)


def between_weather_rows(Z, weather, cls):
    """sqrt(n) * (mean_{c,w} - mean_c) for every class c and weather w."""
    rows = []
    for c in np.unique(cls):
        mc = cls == c
        mu_c = Z[mc].mean(0)
        for w in np.unique(weather[mc]):
            m = mc & (weather == w)
            if m.sum() < 10:
                continue
            rows.append(np.sqrt(m.sum()) * (Z[m].mean(0) - mu_c))
    return np.array(rows)


def identity_rows(Z, video, inst, cls, min_obs=3):
    """sqrt(n_i) * (mean_i - mean_{video,class}): stable identity signal."""
    rows = []
    key_vc = np.char.add(np.char.add(video.astype(str), '|'), cls.astype(str))
    for vc in np.unique(key_vc):
        m = key_vc == vc
        ids = inst[m]
        zz = Z[m]
        u, cnt = np.unique(ids, return_counts=True)
        u = u[cnt >= min_obs]
        if len(u) < 2:
            continue
        means = np.array([zz[ids == i].mean(0) for i in u])
        ns = np.array([(ids == i).sum() for i in u])
        mu = (means * ns[:, None]).sum(0) / ns.sum()
        rows.append(np.sqrt(ns)[:, None] * (means - mu))
    return np.concatenate(rows, 0) if rows else np.zeros((0, Z.shape[1]))


def temporal_rows(Z, video, inst, min_obs=3):
    key = np.char.add(np.char.add(video.astype(str), '|'), inst.astype(str))
    out = []
    for k in np.unique(key):
        m = key == k
        if m.sum() >= min_obs:
            out.append(Z[m] - Z[m].mean(0))
    return np.concatenate(out, 0) if out else np.zeros((0, Z.shape[1]))


def energy(rows, E):
    if rows is None or len(rows) == 0:
        return float('nan')
    tot = (rows ** 2).sum()
    return float(((rows @ E) ** 2).sum() / (tot + EPS))


def principal_angles_deg(A, B):
    try:
        from scipy.linalg import subspace_angles
        return np.degrees(np.sort(subspace_angles(A, B)))
    except Exception:
        s = np.linalg.svd(orth(A).T @ orth(B), compute_uv=False)
        return np.degrees(np.arccos(np.clip(s, -1, 1)))[::-1]


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def weather_probe(Xfit, yfit, Xte, yte, seed=0, max_fit=20000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.RandomState(seed)
    if len(Xfit) > max_fit:
        idx = rng.choice(len(Xfit), max_fit, replace=False)
        Xfit, yfit = Xfit[idx], yfit[idx]
    mu, sd = Xfit.mean(0), Xfit.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs',
                             class_weight='balanced')   # multinomial by default
    clf.fit((Xfit - mu) / sd, yfit)
    return float(balanced_accuracy_score(yte, clf.predict((Xte - mu) / sd)))


def average_precision(scores, pos):
    order = np.argsort(-scores)
    p = pos[order]
    if p.sum() == 0:
        return np.nan
    hits = np.cumsum(p)
    prec = hits / (np.arange(len(p)) + 1.0)
    return float((prec * p).sum() / p.sum())


def build_retrieval(video, frame, inst, gap, window, max_q, seed=0):
    """Within-video re-identification: query at frame f, gallery in
    (f+gap, f+gap+window]. Returns list of (query_idx, gallery_idx, pos)."""
    rng = np.random.RandomState(seed)
    tasks = []
    for v in np.unique(video):
        idx = np.where(video == v)[0]
        fr = frame[idx]
        for q in idx:
            g = idx[(fr > frame[q] + gap) & (fr <= frame[q] + gap + window)]
            if len(g) < 2:
                continue
            pos = inst[g] == inst[q]
            if pos.any() and (~pos).any():
                tasks.append((q, g, pos))
    if len(tasks) > max_q:
        sel = rng.choice(len(tasks), max_q, replace=False)
        tasks = [tasks[i] for i in sorted(sel)]
    return tasks


def build_cross_weather(scene, variant, frame, inst, ref, window, max_q, seed=0):
    """VKITTI2: query in variant w != ref, gallery = ref variant, same scene,
    |frame - f| <= window. Positive = same instance id."""
    rng = np.random.RandomState(seed)
    tasks = []
    for s in np.unique(scene):
        gi = np.where((scene == s) & (variant == ref))[0]
        qi = np.where((scene == s) & (variant != ref))[0]
        for q in qi:
            g = gi[np.abs(frame[gi] - frame[q]) <= window]
            if len(g) < 2:
                continue
            pos = inst[g] == inst[q]
            if pos.any() and (~pos).any():
                tasks.append((q, g, pos))
    if len(tasks) > max_q:
        sel = rng.choice(len(tasks), max_q, replace=False)
        tasks = [tasks[i] for i in sorted(sel)]
    return tasks


def retrieval_map(X, tasks):
    if not tasks:
        return float('nan')
    Xn = normalize(X)
    aps = [average_precision(Xn[g] @ Xn[q], pos) for q, g, pos in tasks]
    return float(np.nanmean(aps))


def shrink(Z, E, beta):
    return normalize(Z - beta * (Z @ E) @ E.T)


def complement(Z, E):
    return Z - (Z @ E) @ E.T


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '--'
    return ('%.' + str(nd) + 'f') % x


def write_tex(path, text):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    classes = [int(c) for c in args.classes.split(',')]
    betas = [float(b) for b in args.betas.split(',')]
    rng = np.random.RandomState(args.seed + 9973)   # separate stream for random bases
    summary = {}

    data, bank = load_label(args.feat_root, args.label, args.domains, classes)
    Z = normalize(data['feat_clean'].astype(np.float64))
    D = Z.shape[1]
    weather = data['variant'] if args.paired else data['weather']
    video, inst, frame = data['video'], data['instance_id'], data['frame_id']
    cls = data['category']
    group = data['scene'] if args.paired else video
    fit = split_mask(group, args.seed)
    te = ~fit
    summary['protocol'] = {
        'args': vars(args), 'rois': len(Z), 'fit_rois': int(fit.sum()),
        'test_rois': int(te.sum()), 'fit_videos': len(np.unique(group[fit])),
        'test_videos': len(np.unique(group[te])),
        'domain_shift_reference': args.ref_domain or 'pooled class mean',
        'weather_probe': 'refitted logistic regression after each transformation',
        'interpretation': 'BDD domain labels also encode scene and identity differences; '
        'these unpaired probes do not identify a pure weather cause.'}
    write_csv(osp.join(args.out, 'csv', 'split_counts.csv'), [
        dict(domain=w, split=s, rois=int((mask & (weather == w)).sum()),
             videos=len(np.unique(group[mask & (weather == w)])))
        for w in np.unique(weather) for s, mask in [('fit', fit), ('test', te)]])
    print('[data] rois=%d fit=%d test=%d D=%d weathers=%s' % (
        len(Z), fit.sum(), te.sum(), D, list(np.unique(weather))))

    # ---------------- subspaces ----------------
    if bank is None or 'pcs' not in bank or not np.abs(bank['pcs']).sum():
        raise SystemExit('checkpoint %s has no initialized ICS bank' % args.label)
    m = args.m or bank['pcs'].shape[0]
    bases = {'ICS (checkpoint)': orth(bank['pcs'][:m].T.astype(np.float64))}
    for lab in args.compare_labels:
        try:
            _, b2 = load_label(args.feat_root, lab, args.domains[:1], classes)
            bases['ICS (%s)' % lab] = orth(b2['pcs'][:m].T.astype(np.float64))
        except SystemExit as e:
            print('[warn] %s' % e)
    rkey = 'feat_' + args.residual_view
    if rkey in data:
        R = normalize(data[rkey].astype(np.float64)) - Z
        ok = np.isfinite(R).all(1)
        bases['FAP-residual PCA'] = top_eig(cov_of(R[fit & ok]), m)
    cc = Z.copy()
    for c in np.unique(cls):
        mm = fit & (cls == c)
        cc[cls == c] -= Z[mm].mean(0)
    bases['Feature PCA'] = top_eig(cov_of(cc[fit], center=False), m)
    bw_fit = between_weather_rows(Z[fit], weather[fit], cls[fit])
    bases['Oracle weather (labels)'] = top_eig(bw_fit.T @ bw_fit, m)
    id_fit = identity_rows(Z[fit], video[fit], inst[fit], cls[fit])
    bases['Oracle identity (labels)'] = top_eig(id_fit.T @ id_fit, m)
    randoms = [orth(rng.randn(D, m)) for _ in range(args.n_random)]
    E = bases['ICS (checkpoint)']

    # ---------------- P1 energy shares (test split) ----------------
    Zt = Z[te]
    q_rows = {
        'Real weather shift': between_weather_rows(Zt, weather[te], cls[te]),
        'Identity': identity_rows(Zt, video[te], inst[te], cls[te]),
        'Temporal (same ID)': temporal_rows(Zt, video[te], inst[te]),
    }
    if args.ref_domain is not None:
        shifts = [shift_rows(Zt, weather[te], cls[te], d, args.ref_domain)
                  for d in args.domains if d != args.ref_domain]
        q_rows['Real weather shift'] = np.concatenate(shifts, axis=0)
    for view, lab in (('fap', 'FAP residual'), ('photo', 'Photometric residual'),
                      ('student', 'Student-view residual')):
        k = 'feat_' + view
        if k in data:
            Rv = normalize(data[k][te].astype(np.float64)) - Zt
            q_rows[lab] = Rv[np.isfinite(Rv).all(1)]
    if args.paired:
        q_rows['Paired real weather'] = paired_rows(Z, data, te, args.ref_variant)
    p1 = []
    for name, Eb in list(bases.items()) + [('Random (mean)', None)]:
        row = {'subspace': name, 'm': m}
        for q, rows in q_rows.items():
            if Eb is None:
                row[q] = float(np.mean([energy(rows, r) for r in randoms]))
            else:
                row[q] = energy(rows, Eb)
        w, i = row['Real weather shift'], row['Identity']
        row['Weather/Identity'] = w / (i + EPS)
        if 'FAP residual' in row:
            row['FAP/Identity'] = row['FAP residual'] / (i + EPS)
        p1.append(row)
    p1.append(dict(subspace='Chance m/D', m=m, **{q: m / float(D) for q in q_rows}))
    write_csv(osp.join(args.out, 'csv', 'P1_energy.csv'), p1)
    summary['P1'] = p1
    if args.ref_domain is not None:
        by_domain = []
        for d in args.domains:
            if d == args.ref_domain:
                continue
            rows = shift_rows(Zt, weather[te], cls[te], d, args.ref_domain)
            for name, Eb in list(bases.items()) + [('Random (mean)', None)]:
                share = (float(np.mean([energy(rows, r) for r in randoms]))
                         if Eb is None else energy(rows, Eb))
                by_domain.append(dict(domain=d, reference=args.ref_domain,
                                      subspace=name, energy_share=share))
        write_csv(osp.join(args.out, 'csv', 'P1_weather_by_domain.csv'), by_domain)
    ang = {n: principal_angles_deg(E, b).tolist() for n, b in bases.items()}
    summary['angles_to_ICS_deg'] = ang
    write_csv(osp.join(args.out, 'csv', 'P1_angles.csv'),
              [dict(subspace=n, **{'angle%d' % (j + 1): a for j, a in enumerate(v)})
               for n, v in ang.items()])

    # ---------------- P2 double dissociation ----------------
    wlab = np.unique(weather, return_inverse=True)[1]
    tasks = build_retrieval(video[te], frame[te], inst[te], args.gap, args.window,
                            args.max_queries, args.seed)
    p2 = []
    spaces = [('E only (%d-d)' % m, lambda X: X @ E),
              ('E-perp (%d-d)' % (D - m), lambda X: complement(X, E)),
              ('Full (%d-d)' % D, lambda X: X)]
    ri = bases['Oracle identity (labels)']
    spaces.append(('Random %d-d' % m, lambda X: X @ randoms[0]))
    spaces.append(('Identity oracle %d-d' % m, lambda X: X @ ri))
    for name, f in spaces:
        Xf, Xt = f(Z[fit]), f(Z[te])
        p2.append({'space': name,
                   'weather_bacc': weather_probe(Xf, wlab[fit], Xt, wlab[te], args.seed),
                   'chance': 1.0 / len(np.unique(wlab)),
                   'id_mAP': retrieval_map(Xt, tasks)})
    write_csv(osp.join(args.out, 'csv', 'P2_dissociation.csv'), p2)
    summary['P2'] = p2
    summary['n_retrieval_queries'] = len(tasks)

    # ---------------- P3 intervention ----------------
    xw_tasks = []
    if args.paired:
        xw_tasks = build_cross_weather(data['scene'][te], data['variant'][te],
                                       frame[te], inst[te], args.ref_variant,
                                       args.window, args.max_queries, args.seed)
    p3 = []
    baseline = intervene(Z, E, 0.0, fit, te, wlab, weather, cls, tasks,
                         xw_tasks, args.seed)
    for sname, Eb in (('ICS', E), ('Random', None),
                      ('Identity oracle', ri)):
        for b in betas:
            if b == 0:
                # All subspaces have exactly the same zero-dose features.
                res = dict(baseline)
            elif Eb is None:
                vals = []
                for r in randoms[:10]:
                    vals.append(intervene(Z, r, b, fit, te, wlab, weather, cls,
                                          tasks, xw_tasks, args.seed))
                res = {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
            else:
                res = intervene(Z, Eb, b, fit, te, wlab, weather, cls, tasks,
                                xw_tasks, args.seed)
            res.update(subspace=sname, beta=b)
            p3.append(res)
            print('[P3] %-16s beta=%.2f %s' % (sname, b, json.dumps(
                {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})))
    write_csv(osp.join(args.out, 'csv', 'P3_intervention.csv'), p3)
    summary['P3'] = p3

    # ---------------- P4 dose response ----------------
    p4 = dose_response(data, Z, E, te)
    write_csv(osp.join(args.out, 'csv', 'P4_dose.csv'), p4)
    summary['P4'] = p4

    # ---------------- P5 stage evolution ----------------
    p5, p5_ang = [], []
    if args.stage_labels:
        prev = None
        for k, lab in enumerate(args.stage_labels):
            try:
                dk, bk = load_label(args.feat_root, lab, args.domains, classes)
            except SystemExit as e:
                print('[warn] %s' % e)
                continue
            if bk is None or not np.abs(bk['pcs']).sum():
                print('[warn] %s: ICS bank not initialized' % lab)
                continue
            Ek = orth(bk['pcs'][:m].T.astype(np.float64))
            Zk = normalize(dk['feat_clean'].astype(np.float64))
            wk = dk['variant'] if args.paired else dk['weather']
            row = {'stage': k + 1, 'label': lab,
                   'adapted_domain': args.stage_domains[k]
                   if k < len(args.stage_domains) else ''}
            for d in args.domains:
                if args.ref_domain is None or d == args.ref_domain:
                    continue
                row['shift:' + d] = energy(
                    shift_rows(Zk, wk, dk['category'], d, args.ref_domain), Ek)
                if 'feat_fap' in dk:
                    md = wk == d
                    Rk = normalize(dk['feat_fap'][md].astype(np.float64)) - Zk[md]
                    row['fap:' + d] = energy(Rk[np.isfinite(Rk).all(1)], Ek)
            p5.append(row)
            if prev is not None:
                p5_ang.append({'from': prev[0], 'to': lab, 'max_angle_deg':
                               float(principal_angles_deg(prev[1], Ek)[-1])})
            prev = (lab, Ek)
        write_csv(osp.join(args.out, 'csv', 'P5_stage_shift_energy.csv'), p5)
        write_csv(osp.join(args.out, 'csv', 'P5_stage_angles.csv'), p5_ang)
    summary['P5'] = p5
    summary['P5_angles'] = p5_ang

    # ---------------- tables and figures ----------------
    tables(args, p1, p2, p3, m, D)
    try:
        figures(args, data, Z, E, bases, randoms, weather, video, inst, te, p1, p4, p5)
    except Exception as e:
        with open(osp.join(args.out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        raise RuntimeError('figure generation failed: %r' % (e,))
    with open(osp.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print('[done] wrote %s' % args.out)


def paired_rows(Z, data, te, ref):
    key = np.char.add(np.char.add(data['scene'].astype(str), '|'),
                      np.char.add(np.char.add(data['frame_id'].astype(str), '|'),
                                  data['instance_id'].astype(str)))
    ref_idx = {}
    for i in np.where(te & (data['variant'] == ref))[0]:
        ref_idx[key[i]] = i
    rows = [Z[i] - Z[ref_idx[key[i]]] for i in np.where(te & (data['variant'] != ref))[0]
            if key[i] in ref_idx]
    print('[paired] %d matched (object, frame) pairs vs %s' % (len(rows), ref))
    return np.array(rows) if rows else np.zeros((0, Z.shape[1]))


def shift_rows(Z, weather, cls, d, ref):
    rows = []
    for c in np.unique(cls):
        a, b = (cls == c) & (weather == d), (cls == c) & (weather == ref)
        if a.sum() >= 10 and b.sum() >= 10:
            rows.append(np.sqrt(min(a.sum(), b.sum())) * (Z[a].mean(0) - Z[b].mean(0)))
    return np.array(rows) if rows else np.zeros((0, Z.shape[1]))


def weather_gap(Z, weather, cls):
    gaps = []
    for c in np.unique(cls):
        mus = [Z[(cls == c) & (weather == w)].mean(0) for w in np.unique(weather)
               if ((cls == c) & (weather == w)).sum() >= 10]
        for i in range(len(mus)):
            for j in range(i + 1, len(mus)):
                gaps.append(np.linalg.norm(mus[i] - mus[j]))
    return float(np.mean(gaps)) if gaps else float('nan')


def intervene(Z, E, beta, fit, te, wlab, weather, cls, tasks, xw_tasks, seed):
    Zb = shrink(Z, E, beta)
    out = {'weather_bacc': weather_probe(Zb[fit], wlab[fit], Zb[te], wlab[te], seed),
           'weather_gap': weather_gap(Zb[te], weather[te], cls[te]),
           'id_mAP': retrieval_map(Zb[te], tasks)}
    if xw_tasks:
        out['cross_weather_mAP'] = retrieval_map(Zb[te], xw_tasks)
    return out


def dose_response(data, Z, E, te):
    rows = []
    fams = {}
    for k in data:
        if k.startswith('feat_') and '_' in k[5:]:
            fam, lvl = k[5:].split('_', 1)
            if fam in ('dark', 'fog', 'fapk'):
                fams.setdefault(fam, []).append((float(lvl.replace('p', '.')), k))
    # Match the identity control to the frames used for the dose views.
    dose_available = np.zeros(len(Z), dtype=bool)
    for levels in fams.values():
        for _, k in levels:
            dose_available |= np.isfinite(data[k]).all(1)
    ref_mask = te & dose_available
    # identity reference: two different objects in the same frame
    # COCO image ids are local to each domain JSON; namespace by video.
    image_key = np.char.add(np.char.add(data['video'][ref_mask].astype(str), '|'),
                            data['image_id'][ref_mask].astype(str))
    ref = identity_pairs(Z[ref_mask], image_key, data['instance_id'][ref_mask])
    ref_in = float(np.mean(np.linalg.norm(ref @ E, axis=1))) if len(ref) else np.nan
    ref_out = float(np.mean(np.linalg.norm(complement(ref, E), axis=1))) if len(ref) else np.nan
    for fam, lv in sorted(fams.items()):
        for level, k in sorted(lv):
            X = data[k][te].astype(np.float64)
            ok = np.isfinite(X).all(1)
            if ok.sum() < 10:
                continue
            Dlt = normalize(X[ok]) - Z[te][ok]
            strength = (1 - level) if fam in ('dark', 'fapk') else level
            rows.append({'family': fam, 'level': level, 'strength': strength,
                         'n': int(ok.sum()),
                         'n_identity_reference_pairs': len(ref),
                         'in_E_norm': float(np.mean(np.linalg.norm(Dlt @ E, axis=1))),
                         'out_E_norm': float(np.mean(np.linalg.norm(complement(Dlt, E), axis=1))),
                         'share_in_E': energy(Dlt, E),
                         'identity_ref_in_E_norm': ref_in,
                         'identity_ref_out_E_norm': ref_out})
    return rows


def identity_pairs(Z, image_id, inst, max_pairs=20000, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for im in np.unique(image_id):
        idx = np.where(image_id == im)[0]
        if len(idx) < 2:
            continue
        a, b = rng.choice(idx, 2, replace=False)
        if inst[a] != inst[b]:
            out.append(Z[a] - Z[b])
        if len(out) >= max_pairs:
            break
    return np.array(out) if out else np.zeros((0, Z.shape[1]))


# --------------------------------------------------------------------------- #
def tables(args, p1, p2, p3, m, D):
    qs = [q for q in ('Real weather shift', 'Paired real weather', 'FAP residual',
                      'Identity', 'Weather/Identity') if q in p1[0]]
    head = ' & '.join(['Subspace ($m{=}%d$)' % m] + [q.replace('/', '\\,/\\,') for q in qs])
    lines = ['\\begin{tabular}{l' + 'c' * len(qs) + '}', '\\toprule', head + ' \\\\',
             '\\midrule']
    for r in p1:
        lines.append(' & '.join([r['subspace'].replace('_', r'\_')] + [
            fmt(r.get(q), 1 if q == 'Weather/Identity' else 3) for q in qs]) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_tex(osp.join(args.out, 'tables', 'tab_P1_energy.tex'), '\n'.join(lines) + '\n')

    lines = ['\\begin{tabular}{lcc}', '\\toprule',
             'Features & Weather probe (bal.\\ acc.) & ID retrieval mAP \\\\', '\\midrule']
    for r in p2:
        lines.append('%s & %s & %s \\\\' % (r['space'], fmt(r['weather_bacc']),
                                            fmt(r['id_mAP'])))
    lines += ['\\midrule', 'Chance & %s & -- \\\\' % fmt(p2[0]['chance']),
              '\\bottomrule', '\\end{tabular}']
    write_tex(osp.join(args.out, 'tables', 'tab_P2_dissociation.tex'), '\n'.join(lines) + '\n')

    cols = ['weather_bacc', 'weather_gap', 'id_mAP'] + (
        ['cross_weather_mAP'] if any('cross_weather_mAP' in r for r in p3) else [])
    lines = ['\\begin{tabular}{lc' + 'c' * len(cols) + '}', '\\toprule',
             'Shrunk subspace & $\\beta$ & ' + ' & '.join(c.replace('_', ' ') for c in cols)
             + ' \\\\', '\\midrule']
    for r in p3:
        lines.append('%s & %.2f & %s \\\\' % (r['subspace'], r['beta'],
                                              ' & '.join(fmt(r.get(c)) for c in cols)))
    lines += ['\\bottomrule', '\\end{tabular}']
    write_tex(osp.join(args.out, 'tables', 'tab_P3_intervention.tex'), '\n'.join(lines) + '\n')


def figures(args, data, Z, E, bases, randoms, weather, video, inst, te, p1, p4, p5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 11, 'xtick.labelsize': 9,
                         'ytick.labelsize': 9, 'pdf.fonttype': 42})
    fdir = osp.join(args.out, 'figs')
    os.makedirs(fdir, exist_ok=True)
    review_pdf = PdfPages(osp.join(fdir, 'all_figures.pdf'))
    rng = np.random.RandomState(args.seed)
    wnames = [w for w in args.domains if w in set(weather)] or list(np.unique(weather))
    if args.paired:
        wnames = list(np.unique(weather))
    cmap = plt.get_cmap('tab10')
    wcol = {w: cmap(i % 10) for i, w in enumerate(wnames)}
    display = {'clear_holdout_45v': 'Clear (holdout)', 'overcast': 'Overcast',
               'rainy': 'Rainy', 'dawn_dusk': 'Dawn/Dusk', 'night': 'Night'}
    pretty = lambda s: display.get(s, s)

    def save(fig, name, rect=None):
        fig.tight_layout(rect=rect)
        fig.savefig(osp.join(fdir, name + '.pdf'), bbox_inches='tight')
        fig.savefig(osp.join(fdir, name + '.png'), dpi=200, bbox_inches='tight')
        review_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ---- Fig. 1: 2x2 (E-plane / complement t-SNE) x (weather / identity)
    ti = np.where(te)[0]
    sub = rng.choice(ti, min(4000, len(ti)), replace=False)
    key = np.char.add(np.char.add(video.astype(str), '|'), inst.astype(str))
    tracks = []
    for w in wnames:                               # 2 long tracks per weather
        kk, cnt = np.unique(key[sub][weather[sub] == w], return_counts=True)
        tracks += list(kk[np.argsort(-cnt)[:2]])
    tcol = {t: plt.get_cmap('tab20')(i * 2 % 20) for i, t in enumerate(tracks)}
    A = Z[sub] @ E
    C = complement(Z[sub], E)
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        Cp = PCA(n_components=min(50, C.shape[1]), random_state=0).fit_transform(C)
        T = TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(Cp)
        complement_method = 't-SNE'
    except Exception as e:
        print('[warn] t-SNE failed (%r); using PCA' % (e,))
        T = C @ top_eig(cov_of(C), 2)
        complement_method = 'PCA'
    fig, ax = plt.subplots(2, 2, figsize=(10.4, 8.2))
    for col, (P, ttl) in enumerate(((A, 'ICS subspace $(e_1,e_2)$'),
                                    (T, 'Complement (%s)' % complement_method))):
        a = ax[0, col]
        for w in wnames:
            mw = weather[sub] == w
            a.scatter(P[mw, 0], P[mw, 1], s=4, alpha=.5, color=wcol[w], label=pretty(w))
        a.set_title(ttl + ', colored by weather', fontsize=9)
        a = ax[1, col]
        a.scatter(P[:, 0], P[:, 1], s=2, color='0.85')
        for t in tracks:
            mt = key[sub] == t
            a.scatter(P[mt, 0], P[mt, 1], s=9, color=tcol[t])
        a.set_title(ttl + ', colored by identity', fontsize=9)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, markerscale=3, fontsize=9, loc='upper center',
               ncol=len(wnames), frameon=False)
    fig.legend([Line2D([], [], marker='o', ls='', color=tcol[t]) for t in tracks],
               ['ID %d' % (i + 1) for i in range(len(tracks))],
               loc='lower center', ncol=5, fontsize=8, frameon=False,
               title='Highlighted tracks (video + instance)')
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    save(fig, 'fig1_quadrant', rect=(0, .11, 1, .95))

    # ---- Fig. 2: perturbation arrows in (e1, top complement PC)
    c1 = top_eig(cov_of(complement(Z[te], E)), 1)[:, 0]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    if args.paired:
        arrows = paired_arrows(Z, data, te, args.ref_variant, rng)
        for v, (a0, a1) in arrows.items():
            ax.quiver(a0 @ E[:, 0], a0 @ c1, (a1 - a0) @ E[:, 0], (a1 - a0) @ c1,
                      angles='xy', scale_units='xy', scale=1, width=.004,
                      color=wcol.get(v, 'k'), label='%s -> %s' % (args.ref_variant, v))
            ax.update_datalim(np.column_stack((a1 @ E[:, 0], a1 @ c1)))
        ax.legend(fontsize=7)
    elif 'feat_fap' in data:
        idx = rng.choice(ti, min(60, len(ti)), replace=False)
        a0 = Z[idx]
        a1 = normalize(data['feat_fap'][idx].astype(np.float64))
        ax.quiver(a0 @ E[:, 0], a0 @ c1, (a1 - a0) @ E[:, 0], (a1 - a0) @ c1,
                  angles='xy', scale_units='xy', scale=1, width=.004,
                  color=[wcol[w] for w in weather[idx]])
        ax.set_title('clean $\\rightarrow$ FAP, same object', fontsize=9)
        ax.scatter(a0 @ E[:, 0], a0 @ c1, s=8,
                   c=[wcol[w] for w in weather[idx]], zorder=3)
        ax.update_datalim(np.column_stack((a1 @ E[:, 0], a1 @ c1)))
        ax.legend([Line2D([], [], color=wcol[w], marker='o', ls='') for w in wnames],
                  [pretty(w) for w in wnames], fontsize=7, loc='upper left',
                  bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_xlabel('$e_1$ (ICS)')
    ax.set_ylabel('top complement direction')
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='datalim')
    save(fig, 'fig2_arrows')

    # ---- Fig. 3: crops at both ends of e1
    if args.montage:
        montage(args, data, Z, E, te, fdir, plt, review_pdf)

    # ---- Fig. 4: dose response
    if p4:
        fams = sorted(set(r['family'] for r in p4))
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.7))
        names = {'dark': 'darkening', 'fog': 'fog blend', 'fapk': 'FAP masking'}
        for i, fam in enumerate(fams):
            rr = sorted([r for r in p4 if r['family'] == fam], key=lambda r: r['strength'])
            x = [r['strength'] for r in rr]
            ax[0].plot(x, [r['in_E_norm'] for r in rr], '-o', color=cmap(i),
                       label=names.get(fam, fam) + ' (in $E$)')
            ax[0].plot(x, [r['out_E_norm'] for r in rr], '--o', color=cmap(i),
                       alpha=.6, label=names.get(fam, fam) + ' (in $E^\\perp$)')
            ax[1].plot(x, [r['share_in_E'] for r in rr], '-o', color=cmap(i),
                       label=names.get(fam, fam))
        ax[0].axhline(p4[0]['identity_ref_in_E_norm'], color='k', lw=1,
                      label='other object (in $E$)')
        ax[0].axhline(p4[0]['identity_ref_out_E_norm'], color='k', lw=1, ls='--',
                      label='other object (in $E^\\perp$)')
        m = E.shape[1]
        ax[1].axhline(m / float(Z.shape[1]), color='0.5', ls=':', label='chance $m/D$')
        ax[0].set_xlabel('perturbation strength')
        ax[1].set_xlabel('perturbation strength')
        ax[0].set_ylabel('mean embedding change')
        ax[1].set_ylabel('share of change inside $E$')
        handles, labels = ax[0].get_legend_handles_labels()
        chance_handles, chance_labels = ax[1].get_legend_handles_labels()
        fig.legend(handles + chance_handles[-1:], labels + chance_labels[-1:],
                   fontsize=8, loc='upper center', ncol=3, frameon=False)
        ax[1].set_ylim(bottom=0)
        save(fig, 'fig4_dose_response', rect=(0, 0, 1, .79))

    # ---- Fig. 5: energy bars
    qs = [q for q in ('Real weather shift', 'Paired real weather', 'FAP residual',
                      'Photometric residual', 'Identity') if q in p1[0]]
    subs = [r for r in p1 if r['subspace'] != 'Chance m/D']
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    wbar = 0.8 / len(subs)
    for i, r in enumerate(subs):
        ax.bar(np.arange(len(qs)) + i * wbar, [r[q] for q in qs], wbar, label=r['subspace'])
    ax.axhline(subs[0]['m'] / float(Z.shape[1]), color='k', ls=':', lw=1, label='chance')
    ax.set_xticks(np.arange(len(qs)) + 0.4 - wbar / 2)
    labels = [('Domain shift\n(relative to Clear)' if getattr(args, 'ref_domain', None)
               else 'Domain mean shift') if q == 'Real weather shift' else q for q in qs]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('energy share in subspace')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=3, loc='upper center', frameon=False)
    ax.set_ylim(bottom=0)
    save(fig, 'fig5_energy_bars', rect=(0, 0, 1, .82))

    # ---- Fig. 6: stage x domain heatmap
    if p5:
        cols = [k for k in p5[0] if k.startswith('shift:')]
        M = np.array([[r.get(c, np.nan) for c in cols] for r in p5])
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        im = ax.imshow(M, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([pretty(c[6:]) for c in cols], fontsize=10)
        ax.set_yticks(range(len(p5)))
        ax.set_yticklabels(['After %s' % pretty(r['adapted_domain']) for r in p5], fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt(M[i, j], 3), ha='center', va='center',
                        color='black' if M[i, j] > .5 else 'white', fontsize=10)
        fig.colorbar(im, ax=ax, label='shift energy in ICS subspace')
        save(fig, 'fig6_stage_heatmap')
    page_count = review_pdf.get_pagecount()
    review_pdf.close()
    with open(osp.join(args.out, 'figure_manifest.json'), 'w') as stream:
        json.dump(dict(combined_pdf='figs/all_figures.pdf', page_count=page_count,
                       complement_visualization=complement_method), stream, indent=2)


def paired_arrows(Z, data, te, ref, rng, n=12):
    key = np.char.add(np.char.add(data['scene'].astype(str), '|'),
                      np.char.add(np.char.add(data['frame_id'].astype(str), '|'),
                                  data['instance_id'].astype(str)))
    ref_idx = {key[i]: i for i in np.where(te & (data['variant'] == ref))[0]}
    out = {}
    for v in np.unique(data['variant']):
        if v == ref:
            continue
        cand = [i for i in np.where(te & (data['variant'] == v))[0] if key[i] in ref_idx]
        if not cand:
            continue
        sel = rng.choice(cand, min(n, len(cand)), replace=False)
        out[v] = (Z[[ref_idx[key[i]] for i in sel]], Z[sel])
    return out


def montage(args, data, Z, E, te, fdir, plt, review_pdf=None):
    try:
        import cv2
        read = lambda p: cv2.imread(p)[:, :, ::-1]
    except Exception:
        from matplotlib.image import imread as read
    ti = np.where(te)[0]
    records = []
    names = {'clear_holdout_45v': 'Clear', 'overcast': 'Overcast',
             'rainy': 'Rainy', 'dawn_dusk': 'Dawn/Dusk', 'night': 'Night'}
    for j in range(E.shape[1]):
        proj = Z[ti] @ E[:, j]
        order = np.argsort(proj)
        picks = []
        for side in (order[:400], order[::-1][:400]):
            seen, chosen = set(), []
            for k in side:
                i = ti[k]
                kk = (data['video'][i], int(data['instance_id'][i]))
                if kk in seen:
                    continue
                seen.add(kk)
                chosen.append(i)
                if len(chosen) == 10:
                    break
            picks.append(chosen)
        fig, ax = plt.subplots(2, 10, figsize=(14, 3.6))
        for r, chosen in enumerate(picks):
            for c in range(10):
                a = ax[r, c]
                a.axis('off')
                if c >= len(chosen):
                    continue
                i = chosen[c]
                try:
                    img = read(str(data['img_path'][i]))
                    x0, y0, x1, y1 = [int(round(v)) for v in data['bbox'][i]]
                    a.imshow(img[max(y0, 0):y1, max(x0, 0):x1])
                except Exception:
                    continue
                domain = str(data['weather'][i])
                projection = float(Z[i] @ E[:, j])
                a.set_title('%s\n%+.3f' % (names.get(domain, domain), projection), fontsize=8)
                records.append(dict(axis=j + 1, side='low' if r == 0 else 'high',
                                    column=c + 1, domain=domain, projection=projection,
                                    video=str(data['video'][i]),
                                    instance_id=int(data['instance_id'][i]),
                                    frame_id=int(data['frame_id'][i]),
                                    img_path=str(data['img_path'][i]),
                                    bbox=data['bbox'][i].tolist()))
        ax[0, 0].text(-0.3, 0.5, 'low $e_%d$' % (j + 1), transform=ax[0, 0].transAxes,
                      ha='right', fontsize=8)
        ax[1, 0].text(-0.3, 0.5, 'high $e_%d$' % (j + 1), transform=ax[1, 0].transAxes,
                      ha='right', fontsize=8)
        fig.tight_layout()
        fig.savefig(osp.join(fdir, 'fig3_montage_e%d.pdf' % (j + 1)))
        fig.savefig(osp.join(fdir, 'fig3_montage_e%d.png' % (j + 1)), dpi=200)
        if review_pdf is not None:
            review_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    write_csv(osp.join(args.out, 'csv', 'figure3_crop_manifest.csv'), records)


if __name__ == '__main__':
    main()
