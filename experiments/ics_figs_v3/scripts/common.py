"""CPU-only readers; every output is confined to this experiment package."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EXPERIMENTS = REPO / 'experiments'
IDAPR = EXPERIMENTS / 'idapr'
EVIDENCE = EXPERIMENTS / 'ics_evidence'
TOOLS = REPO / 'tools/analysis'
IDAPR_SOURCE = IDAPR / 'source/tools/analysis'
EVIDENCE_SCRIPTS = EVIDENCE / 'scripts'
DOMAINS = ['clear_holdout_45v', 'overcast', 'rainy', 'dawn_dusk', 'night']
TARGETS = DOMAINS[1:]
DISPLAY = dict(zip(DOMAINS, ['Clear', 'Overcast', 'Rainy', 'Dawn/Dusk', 'Night']))
KS = [2, 8, 16]
BASE_COMMIT = '6375023aef7987a66e2d75809f85c9a551f8afa2'

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only execution required'
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    assert os.environ.get(name) == '4', (name, os.environ.get(name))
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
sys.path[:0] = [str(IDAPR_SOURCE), str(EVIDENCE_SCRIPTS), str(TOOLS)]

import numpy as np
import ics_probe_analyze as p
import ics_probe_drift as drift
from idapr_basis import label_free_bases


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verify(records):
    for record in records:
        assert sha(record['path']) == record['sha256'], ('input changed', record['path'])


def hf_root():
    values = {}
    for line in (IDAPR / 'inputs.txt').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    root = Path(values['HF_feat_root'])
    assert root.is_dir(), ('HF feature root unavailable; no F fallback', str(root))
    return root


def feature_preflight(root, stages):
    rows = []
    for stage in stages:
        required = ['feat_clean', 'feat_fap'] if stage == 1 else ['feat_clean']
        if stage == 4:
            required += ['feat_fap', 'feat_photo', 'feat_student']
        for domain in DOMAINS:
            path = root / ('HF_stage%d_e5' % stage) / (domain + '.npz')
            assert path.is_file(), ('missing HF input; never use F', str(path))
            with np.load(str(path), allow_pickle=False) as arrays:
                assert all(key in arrays.files for key in required), (str(path), required)
                row = dict(model='HF', stage=stage, domain=domain, **fingerprint(path))
                for key in required:
                    value = arrays[key]
                    assert value.ndim == 2 and value.shape[1] == 256
                    assert np.isfinite(value).all(), (str(path), key)
                    row[key + '_shape'] = list(value.shape)
                assert np.array_equal(np.unique(arrays['category']), [1]), str(path)
                row['categories'] = [1]
                rows.append(row)
    return rows


def save_basis(name, E):
    assert np.isfinite(E).all() and E.shape[0] == 256
    assert np.allclose(E.T @ E, np.eye(E.shape[1]), atol=1e-9)
    path = ROOT / 'bases' / (name + '.npy')
    path.parent.mkdir(exist_ok=True)
    np.save(str(path), E)
    return fingerprint(path)
