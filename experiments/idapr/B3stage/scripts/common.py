"""Read-only experiment inputs and local output helpers (Python 3.7)."""
import os, sys, json, hashlib, csv, copy, random
from pathlib import Path
Q = Path(__file__).resolve().parents[1]
RUN = Path('/data1/liuyeqiang/damot_rebuttal_20261026')
HF = Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
CLASSES = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
INPUTS = json.loads((Q/'manifests/inputs.json').read_text())
def setup(label):
    code = Q/('code_hf' if label == 'HF' else 'code')
    sys.path.insert(0, str(code))
    import torch
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(0)
    import darth.models, darth.datasets
    return code
def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str,ensure_ascii=False));tmp.replace(p)
def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(p)
def strip_init(x):
    if isinstance(x,dict):
        x.pop('init_cfg',None);x.pop('pretrained',None)
        for v in x.values():strip_init(v)
    elif isinstance(x,(list,tuple)):
        for v in x:strip_init(v)
def seed(n):
    import numpy as np, torch
    random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def checkpoint(label):
    return Path(INPUTS['F_stage4'] if label=='F' else INPUTS['HF_stage4_epoch5'])
def config(label):
    from mmcv import Config
    return Config.fromfile(INPUTS['F_adaptation_config'] if label=='F' else INPUTS['HF_base_config'])
def training_data(cfg,domain):
    ds=copy.deepcopy(cfg.data.train)
    def fix(d):
        if isinstance(d,list):
            for v in d:fix(v)
        elif isinstance(d,dict):
            if 'ann_file' in d:
                d['ann_file']=str(RUN/'data/annotations'/(domain+'.json'))
                local_views=Q/'data_views'/domain
                d['img_prefix']=str(local_views if local_views.is_dir() else RUN/'data/views'/domain)+'/'
                d['classes']=CLASSES
            elif 'dataset' in d:fix(d['dataset'])
    fix(ds);return ds
def install_checkpointing():
    import torch.utils.checkpoint as cp
    orig=cp.checkpoint
    def call(function,*args,**kwargs):
        kwargs['use_reentrant']=False
        return orig(function,*args,**kwargs)
    cp.checkpoint=call
