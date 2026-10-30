"""Frozen inputs and isolated inference-trim outputs; no training is launched."""
import csv,hashlib,json,os,sys,random,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ALIAS=Path('/data1/liuyeqiang/ics_infer_trim_20261031_022136')
CODE=ROOT/'code';sys.path.insert(0,str(CODE))
PY='/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python'
CROOT=Path('/data1/liuyeqiang/ics_v2_20261030_184026')
ADV=Path('/data1/liuyeqiang/ics_advantage_20261031_004629')
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
NEW=Path('/data1/liuyeqiang/damot_hf_intics_20261031_004258')
RUN=Path('/data1/liuyeqiang/damot_rebuttal_20261026')
CLASSES=['pedestrian','car','truck','bus','motorcycle','bicycle']
DOMAINS=['night','dawn_dusk']

def nas_mounted():
    # Read the local mount table; avoid a network stat on every video frame.
    return any(line.split()[4]=='/data1/liuyeqiang/nas/A100' for line in Path('/proc/self/mountinfo').read_text().splitlines())

def read_json(path):
    # Some SSHFS/SFTP rename implementations expose a brief replacement gap.
    for attempt in range(20):
        try:return json.loads(Path(path).read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if attempt==19:raise
            time.sleep(.1)

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def csvout(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def strip_init(v):
    if isinstance(v,dict):
        v.pop('init_cfg',None);v.pop('pretrained',None)
        for x in v.values():strip_init(x)
    elif isinstance(v,(list,tuple)):
        for x in v:strip_init(x)
def checkpoint(model):
    data=json.loads((CROOT/'manifests/final_model_provenance.json').read_text())
    return Path(next(r['path'] for r in data['models'] if r['group']==model))
def seed():
    import torch
    random.seed(20261029);np.random.seed(20261029);torch.manual_seed(20261029);torch.cuda.manual_seed_all(20261029)
def environment(gpu=None):
    # Multiprocessing uses Unix-domain sockets: keep their local path short.
    scratch=Path('/tmp/ics_trim_022136');scratch.mkdir(parents=True,exist_ok=True)
    e=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(scratch))
    e['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
    return e

def cases(model):
    zero=dict(label=model+'_b0',beta=0.,basis_source='checkpoint_bank',npy_path=None)
    if model=='C0':return [zero,dict(label='C0_INT_b025',beta=.25,basis_source='npy',npy_path=str(ROOT/'inputs/C2_basis.npy')),
                          dict(label='C0_RAND_b025',beta=.25,basis_source='npy',npy_path=str(ROOT/'inputs/C4_basis.npy'))]
    return [zero,dict(label=model+'_b025',beta=.25,basis_source='npy',npy_path=str(ROOT/'inputs'/(model+'_basis.npy')))]
