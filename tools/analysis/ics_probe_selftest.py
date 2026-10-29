#!/usr/bin/env python
"""Synthetic self-test for ics_probe_analyze.py (CPU, ~1 min).

Writes fake features with a PLANTED weather subspace W (2-d) and identity
signal outside W, then runs the analysis. The checks below must pass; if they
do not, the analysis code (not the model) is wrong.

    python tools/analysis/ics_probe_selftest.py --out /tmp/ics_selftest
"""
import argparse
import json
import os
import os.path as osp
import subprocess
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))


def make(root, label, rng, W, pcs, domains, D=256, paired=False):
    os.makedirs(osp.join(root, label), exist_ok=True)
    wshift = {d: 0.6 * W @ rng.randn(2) for d in domains}
    ids_per_video, frames = 6, 20
    for d in domains:
        rows = {k: [] for k in ('video', 'scene', 'variant', 'instance_id',
                                'frame_id', 'image_id', 'category', 'img_path')}
        feats = {}
        img = 0
        for v in range(8):
            vname = 'scene%02d' % v if paired else '%s_v%02d' % (d, v)
            ident = rng.randn(ids_per_video, D)
            ident -= (ident @ W) @ W.T               # identity lives outside W
            ident = 1.0 * ident / np.linalg.norm(ident, axis=1, keepdims=True)
            if paired:                               # same objects in every variant
                ident = np.random.RandomState(100 + v).randn(ids_per_video, D)
                ident -= (ident @ W) @ W.T
                ident /= np.linalg.norm(ident, axis=1, keepdims=True)
            for f in range(frames):
                img += 1
                base = 2.0 * np.ones(D) / np.sqrt(D) + ident + wshift[d] + \
                    0.05 * rng.randn(ids_per_video, D)
                views = {'clean': base,
                         'fap': base + 0.3 * (rng.randn(ids_per_video, 2) @ W.T)
                         + 0.02 * rng.randn(ids_per_video, D),
                         'photo': base + 0.2 * (rng.randn(ids_per_video, 2) @ W.T)
                         + 0.05 * rng.randn(ids_per_video, D),
                         'student': base + 0.35 * (rng.randn(ids_per_video, 2) @ W.T)}
                for lvl in ('0p80', '0p60', '0p40', '0p25'):
                    s = 1 - float(lvl.replace('p', '.'))
                    views['dark_' + lvl] = base + s * (np.ones((ids_per_video, 1)) * (W[:, 0] * 1.5))
                for lvl in ('0p15', '0p30', '0p45', '0p60'):
                    a = float(lvl.replace('p', '.'))
                    views['fog_' + lvl] = base + a * (np.ones((ids_per_video, 1)) * (W[:, 1] * 1.5))
                for lvl in ('0p90', '0p70', '0p50', '0p30'):
                    s = 1 - float(lvl.replace('p', '.'))
                    views['fapk_' + lvl] = base + s * (rng.randn(ids_per_video, 2) @ W.T)
                for k, x in views.items():
                    feats.setdefault('feat_' + k, []).append(x.astype(np.float32))
                for i in range(ids_per_video):
                    rows['video'].append(vname if not paired else '%s_%s' % (vname, d))
                    rows['scene'].append(vname)
                    rows['variant'].append(d)
                    rows['instance_id'].append(i)
                    rows['frame_id'].append(f)
                    rows['image_id'].append(img)
                    rows['category'].append(1)
                    rows['img_path'].append('none.jpg')
        arrays = {k: np.concatenate(v, 0) for k, v in feats.items()}
        for k in ('video', 'scene', 'variant', 'img_path'):
            arrays[k] = np.array(rows[k])
        for k in ('instance_id', 'frame_id', 'image_id', 'category'):
            arrays[k] = np.array(rows[k], dtype=np.int64)
        arrays['weather'] = np.array([d] * len(rows['video']))
        arrays['bbox'] = np.zeros((len(rows['video']), 4), np.float32)
        arrays['bank_pcs'] = pcs.astype(np.float32)
        arrays['bank_initialized'] = np.ones(1, bool)
        np.savez_compressed(osp.join(root, label, d + '.npz'), **arrays)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='/tmp/ics_selftest')
    a = p.parse_args()
    rng = np.random.RandomState(0)
    D = 256
    W, _ = np.linalg.qr(rng.randn(D, 2))
    good = (W + 0.1 * rng.randn(D, 2) / np.sqrt(D)).T
    domains = ['clear', 'overcast', 'rainy', 'night']
    feat = osp.join(a.out, 'feats')
    make(feat, 'GOOD', rng, W, good, domains)
    make(feat, 'RANDOMBANK', rng, W, rng.randn(2, D), domains)
    for lab in ('GOOD', 'RANDOMBANK'):
        cmd = [sys.executable, osp.join(HERE, 'ics_probe_analyze.py'),
               '--feat-root', feat, '--label', lab, '--domains'] + domains + [
               '--ref-domain', 'clear', '--stage-labels', 'GOOD', 'RANDOMBANK',
               '--stage-domains', 'overcast', 'rainy', '--n-random', '10',
               '--out', osp.join(a.out, 'res_' + lab)]
        subprocess.check_call(cmd)
    g = json.load(open(osp.join(a.out, 'res_GOOD', 'summary.json')))
    r = json.load(open(osp.join(a.out, 'res_RANDOMBANK', 'summary.json')))
    gi = g['P1'][0]
    ri = r['P1'][0]
    checks = {
        'planted bank captures weather shift (>0.8)': gi['Real weather shift'] > 0.8,
        'planted bank carries little identity (<0.05)': gi['Identity'] < 0.05,
        'random bank near chance (<0.1)': ri['Real weather shift'] < 0.1,
        'random baseline near m/D (<0.05)': [row for row in g['P1'] if row['subspace'] == 'Random (mean)'][0]['Identity'] < 0.05,
        'E-only weather probe > E-perp': g['P2'][0]['weather_bacc'] > g['P2'][1]['weather_bacc'],
        'E-perp ID mAP > E-only': g['P2'][1]['id_mAP'] > g['P2'][0]['id_mAP'],
    }
    for k, v in checks.items():
        print('[%s] %s' % ('PASS' if v else 'FAIL', k))
    # paired (VKITTI2-style) mode
    make(feat, 'PAIRED', rng, W, good, ['clone', 'fog', 'rain'], paired=True)
    subprocess.check_call([sys.executable, osp.join(HERE, 'ics_probe_analyze.py'),
                           '--feat-root', feat, '--label', 'PAIRED', '--paired',
                           '--ref-variant', 'clone', '--domains', 'clone', 'fog', 'rain',
                           '--n-random', '10', '--out', osp.join(a.out, 'res_PAIRED')])
    pr = json.load(open(osp.join(a.out, 'res_PAIRED', 'summary.json')))
    rnd = [row for row in pr['P1'] if row['subspace'] == 'Random (mean)'][0]
    ok2 = pr['P1'][0]['Paired real weather'] > 5 * rnd['Paired real weather']
    print('[%s] paired weather residual: planted bank %.3f vs random %.3f (>5x)' % (
        'PASS' if ok2 else 'FAIL', pr['P1'][0]['Paired real weather'],
        rnd['Paired real weather']))
    sys.exit(0 if all(checks.values()) and ok2 else 1)


if __name__ == '__main__':
    main()
