"""Task-local paths and exact sample/statistics protocol; no network training."""
import csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code'
PY='/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'
CROOT=Path('/data1/liuyeqiang/ics_v2_20261030_184026')
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
RUN=Path('/data1/liuyeqiang/damot_rebuttal_20261026')
NEW=Path('/data1/liuyeqiang/damot_hf_intics_20261031_004258')
DOMAINS=['night','dawn_dusk']
MODELS=['C0','C1','C2','C3','C4']
CLASSES=['pedestrian','car','truck','bus','motorcycle','bicycle']
NBOOT=10000
sys.path.insert(0,str(CODE))
import ics_probe_analyze as probe

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def read_json(path):
    for attempt in range(20):
        try:return json.loads(Path(path).read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if attempt==19:raise
            time.sleep(.1)

def csvout(path,rows,fieldnames=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fieldnames is None:
        if not rows:raise ValueError('Empty CSV rows require explicit fieldnames: '+str(path))
        fieldnames=list(rows[0])
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)

def readcsv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def image_prefix(domain):
    for p in [CROOT/'data_views'/domain,RUN/'data/views'/domain]:
        if p.is_dir():return p
    raise FileNotFoundError(domain)

def checkpoint(model):
    extra=ROOT/'manifests/batch2_model_provenance.json'
    if extra.exists():
        entries=json.loads(extra.read_text())
        if model in entries:return Path(entries[model]['checkpoint'])
    a=json.loads((CROOT/'manifests/final_model_provenance.json').read_text())
    return Path(next(x['path'] for x in a['models'] if x['group']==model))

def env(cpu=False):
    scratch=Path('/tmp/damot_advantage_004629');scratch.mkdir(parents=True,exist_ok=True)
    e=dict(os.environ,OMP_NUM_THREADS='2' if cpu else '1',OPENBLAS_NUM_THREADS='2' if cpu else '1',
           MKL_NUM_THREADS='2' if cpu else '1',MKL_THREADING_LAYER='GNU',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(scratch))
    if cpu:e['CUDA_VISIBLE_DEVICES']=''
    return e
