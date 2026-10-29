#!/usr/bin/env python
"""Local controller; all commands and return codes are retained in progress.json."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]
CODE = RUN / 'code'
PY = str(RUN / 'env/bin/python')
DOMAINS = ['clear_holdout_45v', 'overcast', 'rainy', 'dawn_dusk', 'night']
LABELS = ['F_stage%d_e5' % i for i in range(1, 5)]
LOCK = threading.Lock()
STATE = {'status': 'single_domain_validation', 'jobs': []}


def save():
    STATE['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp = OUT / 'progress.json.tmp'
    temp.write_text(json.dumps(STATE, indent=2))
    temp.replace(OUT / 'progress.json')


def command(args, gpu, name):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', MKL_THREADING_LAYER='GNU', PYTHONUNBUFFERED='1')
    log = OUT / 'logs' / (name + '.log')
    if log.exists():
        raise RuntimeError('refusing to replace log: %s' % log)
    job = dict(name=name, gpu=str(gpu), command=[PY] + args, log=str(log),
               status='running', start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with LOCK:
        STATE['jobs'].append(job)
        save()
    with log.open('w') as stream:
        process = subprocess.Popen([PY] + args, cwd=str(CODE), env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        with LOCK:
            job['pid'] = process.pid
            save()
        code = process.wait()
    with LOCK:
        job.update(returncode=code, status='complete' if code == 0 else 'failed',
                   end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
    if code:
        raise RuntimeError('%s failed, exit %s; see %s' % (name, code, log))


def extract(label, domains, gpu, views, dose):
    args = ['tools/analysis/ics_probe_extract.py', '--checkpoint',
            str(RUN / 'cache/weights' / (label + '.pth')), '--label', label,
            '--views', views, '--dose-frames-per-video', str(dose),
            '--out-dir', str(OUT / 'feats')]
    for d in domains:
        if (OUT / 'feats' / label / (d + '.npz')).exists():
            raise RuntimeError('output already exists: %s/%s' % (label, d))
        args += ['--domain', d, str(RUN / 'data/annotations' / (d + '.json')),
                 str(RUN / 'data/views' / d)]
    command(args, gpu, 'extract_' + label + '_' + '_'.join(domains) + '_verified')


def validate_domain(label, domain):
    path = OUT / 'feats' / label / (domain + '.npz')
    with np.load(str(path), allow_pickle=False) as data:
        n = len(data['category'])
        assert n > 0 and (data['category'] == 1).all(), (label, domain, n)
        assert bool(data['bank_initialized'].any())
        assert int(data['bank_num_updates'].sum()) > 0
        for view in (['clean', 'fap', 'photo', 'student'] if label == LABELS[-1]
                     else ['clean', 'fap']):
            assert data['feat_' + view].shape == (n, 256)
            assert np.isfinite(data['feat_' + view]).all()
        dose_n = 0
        if label == LABELS[-1]:
            dose = data['feat_fapk_0p90']
            ok = np.isfinite(dose).all(1)
            dose_n = int(ok.sum())
            assert dose_n > 0
            assert np.allclose(dose[ok], data['feat_fap'][ok], atol=1e-5)
        return dict(label=label, domain=domain, rois=n, dose_rois=dose_n,
                    bank_num_updates=int(data['bank_num_updates'].sum()))


def stage_worker():
    for label in LABELS[:2]:
        extract(label, DOMAINS, 8, 'clean,fap', 0)


def main():
    save()
    extract(LABELS[-1], ['overcast'], 7, 'clean,fap,photo,student', 3)
    first = validate_domain(LABELS[-1], 'overcast')
    print('[validated single domain] %s' % json.dumps(first), flush=True)
    with LOCK:
        STATE['single_domain_validation'] = first
        STATE['status'] = 'extracting'
        save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(extract, LABELS[-1], [d for d in DOMAINS if d != 'overcast'],
                            7, 'clean,fap,photo,student', 3),
                pool.submit(stage_worker),
                pool.submit(extract, LABELS[2], DOMAINS, 9, 'clean,fap', 0)]
        for future in concurrent.futures.as_completed(jobs):
            future.result()
    checked = []
    for label in LABELS:
        metas = {}
        for d in DOMAINS:
            checked.append(validate_domain(label, d))
            metas[d] = json.loads((OUT / 'feats' / label / (d + '_meta.json')).read_text())
        first_meta = dict(metas[DOMAINS[0]])
        first_meta.update(domains=DOMAINS, per_domain_meta={d: d + '_meta.json' for d in DOMAINS},
                          domain_rois={d: metas[d]['rois'] for d in DOMAINS})
        (OUT / 'feats' / label / 'meta.json').write_text(json.dumps(first_meta, indent=2))
    # Verify that stage comparisons use exactly the same sampled objects.
    for d in DOMAINS:
        with np.load(str(OUT / 'feats' / LABELS[-1] / (d + '.npz'))) as ref:
            for label in LABELS[:-1]:
                with np.load(str(OUT / 'feats' / label / (d + '.npz'))) as data:
                    for key in ['video', 'instance_id', 'frame_id', 'image_id', 'category', 'bbox']:
                        assert np.array_equal(data[key], ref[key]), (label, d, key)
    (OUT / 'feature_validation.json').write_text(json.dumps(checked, indent=2))
    selftest = OUT / 'selftest_corrected/logs/selftest.log'
    if not selftest.exists() or selftest.read_text().count('[PASS]') != 7:
        raise RuntimeError('corrected selftest has not passed all seven checks')
    with LOCK:
        STATE['status'] = 'analyzing'
        save()
    command(['tools/analysis/ics_probe_analyze.py', '--feat-root', str(OUT / 'feats'),
             '--label', LABELS[-1], '--domains'] + DOMAINS +
            ['--ref-domain', DOMAINS[0], '--stage-labels'] + LABELS +
            ['--stage-domains', 'overcast', 'rainy', 'dawn_dusk', 'night',
             '--classes', '1', '--montage', '--out', str(OUT / ('analysis_' + LABELS[-1]))],
            '', 'analyze_' + LABELS[-1])
    figs = OUT / ('analysis_' + LABELS[-1]) / 'figs'
    for name in ['fig1_quadrant', 'fig2_arrows', 'fig3_montage_e1', 'fig3_montage_e2',
                 'fig4_dose_response', 'fig5_energy_bars', 'fig6_stage_heatmap']:
        for suffix in ['.png', '.pdf']:
            assert (figs / (name + suffix)).stat().st_size > 1000
    with LOCK:
        STATE['status'] = 'awaiting_visual_review'
        save()
    print('[done] extraction, analysis and artifact checks complete', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        with LOCK:
            STATE.update(status='failed', error=repr(error))
            save()
        raise
