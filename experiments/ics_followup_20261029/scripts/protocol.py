"""Resolved effective pipelines and explicit configuration changes."""
import copy,json
from pathlib import Path

def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,Path):return str(x)
    return x

def diff(a,b,path=''):
    a,b=plain(a),plain(b)
    if isinstance(a,dict) and isinstance(b,dict):
        out=[]
        for k in sorted(set(a)|set(b)):
            out+=diff(a.get(k,'<missing>'),b.get(k,'<missing>'),path+'.'+k if path else k)
        return out
    return [] if a==b else [dict(field=path,before=a,after=b)]

def views(cfg):
    model=cfg.model;head=model.get('track_head',{})
    train=cfg.data.get('train',[]);train=train if isinstance(train,list) else [train]
    return dict(teacher_pipeline=plain(model.get('teacher_pipeline')),
                student_pipeline=plain(model.get('student_pipeline')),
                contrastive_pipeline=plain(model.get('contrastive_pipeline')),
                data_train=[plain(d) for d in train],
                track_head={k:plain(v) for k,v in head.items() if k.startswith(('causal_','roi_proto_'))})

def write_views(cfg,dest,extra=None):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    value=views(cfg)
    if extra:value['execution_notes']=extra
    dest.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def write_json(dest,value):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n');tmp.replace(dest)
