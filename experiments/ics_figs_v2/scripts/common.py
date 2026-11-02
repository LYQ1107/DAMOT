"""Read-only access to the historical evidence; outputs stay in this package."""
import csv
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent
EVIDENCE = EXPERIMENTS / 'ics_evidence'
SCRIPTS = EVIDENCE / 'scripts'


def configure():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only run required'
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        assert os.environ.get(name) == '4', (name, os.environ.get(name))
    sys.dont_write_bytecode = True
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'matplotlib')
    sys.path.insert(0, str(SCRIPTS))


configure()


@contextmanager
def historical_probe_blas():
    """Replay historical two-thread MKL reductions; restore the requested four.

    part1_F_terminal_transcript.txt records the original two-thread launch.
    Unchanged L-BFGS probes change a few predictions under four-thread MKL.
    Only historical loading/projection/probe numerics use this compatibility
    context; the shell environment and PCA/t-SNE retain the requested four.
    """
    import mkl
    previous = mkl.get_max_threads()
    mkl.set_num_threads(2)
    try:
        yield
    finally:
        mkl.set_num_threads(previous)


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verify_sources(records):
    for record in records:
        assert sha(record['path']) == record['sha256'], ('source changed', record['path'])
