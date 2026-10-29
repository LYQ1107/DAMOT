"""X1: exact partial derivatives of the two association objectives."""
import argparse,time,copy
from common import *
p=argparse.ArgumentParser();p.add_argument('--model',choices=['F','HF'],required=True);p.add_argument('--domain',required=True);p.add_argument('--batches',type=int,default=20);a=p.parse_args()
code=setup(a.model)
import torch,numpy as np
from mmcv.parallel import scatter
from mmtrack.datasets import build_dataloader
from darth.datasets import build_dataset
from darth.models import build_model
cfg=config(a.model);seed(int(cfg.seed));torch.backends.cudnn.benchmark=False
strip_init(cfg.model);install_checkpointing()
cfg.data.train=training_data(cfg,a.domain);cfg.data.samples_per_gpu=16
out=Q/'X1'/a.model/a.domain;out.mkdir(parents=True,exist_ok=True);cfg.dump(str(out/'resolved.py'))
m=build_model(copy.deepcopy(cfg.model));ck=torch.load(str(checkpoint(a.model)),map_location='cpu');m.load_state_dict(ck['state_dict'],strict=True);del ck
# Only these parameters are differentiated. Disabling upstream parameter grads
# leaves the requested partial derivatives and the forward function unchanged.
for n,t in m.named_parameters():t.requires_grad_(n.startswith('track_head.embed_head.'))
m=m.cuda();m.train();m.teacher_detector.eval();m.teacher_track_head.eval()
m.set_current_target_domain(a.domain)
m.roi_proto_start_stage(a.domain)
h=m.track_head;bank=h.causal_bank;capture={};orig=h.extract_bbox_feats
def extract(*args,**kwargs):
    z=orig(*args,**kwargs)
    if 'key' not in capture:capture['key']=z;z.retain_grad()
    return z
h.extract_bbox_feats=extract
params=tuple(h.embed_head.parameters())
ds=build_dataset(cfg.data.train)
loader=build_dataloader(ds,samples_per_gpu=16,workers_per_gpu=int(cfg.data.workers_per_gpu),num_gpus=1,dist=False,shuffle=True,seed=int(cfg.seed),persistent_workers=True)
def scalar(v):
    if isinstance(v,(list,tuple)):return sum(x.mean() for x in v)
    return v.mean()
def cosine(x,y):
    nx=x.norm();ny=y.norm()
    return float((x*y).sum()/(nx*ny)) if nx>0 and ny>0 else float('nan')
def stats(x,y,prefix):
    comb=x+float(h.causal_loss_weight)*y
    return {prefix+'cos_pcl_trim':cosine(x,y),prefix+'norm_ratio':float(y.norm()/x.norm()) if x.norm()>0 else float('nan'),prefix+'cos_total_pcl':cosine(comb,x)}
rows=[]
for i,b in enumerate(loader):
    if i>=a.batches:break
    t=time.time();capture.clear();m.zero_grad(set_to_none=True)
    b=scatter(b,[0])[0]
    before=int(bank.num_updates.item());losses=m.forward_train(**b)
    key_count=int(capture['key'].shape[0]) if 'key' in capture else 0
    row=dict(model=a.model,domain=a.domain,batch=i,triggered=int('loss_track_trim' in losses),initialized=bool(bank.initialized.item()),num_updates_before=before,num_updates_after=int(bank.num_updates.item()),trim_alpha=float(h.causal_trim_alpha),num_pcs=int(h.causal_num_pcs),loss_weight=float(h.causal_loss_weight),key_rois=key_count,key_trim_active=key_count>=int(bank.min_samples),min_samples=int(bank.min_samples))
    if row['triggered']:
        lp=scalar(losses['loss_track'])+scalar(losses['loss_track_aux'])
        lt=(scalar(losses['loss_track_trim'])+scalar(losses['loss_track_aux_trim']))/h.causal_loss_weight
        targets=params+(capture['key'],)
        gp=torch.autograd.grad(lp,targets,retain_graph=True,allow_unused=False)
        gt=torch.autograd.grad(lt,targets,retain_graph=True,allow_unused=False)
        vp=torch.cat([v.detach().reshape(-1) for v in gp[:-1]])
        vt=torch.cat([v.detach().reshape(-1) for v in gt[:-1]])
        row.update(stats(vp,vt,'param_'));row.update(stats(gp[-1].detach(),gt[-1].detach(),'key_'))
        E=torch.linalg.qr(bank.pcs.detach().T,mode='reduced')[0]
        d=gt[-1].detach()-gp[-1].detach();energy=(d*d).sum()
        row['key_diff_energy_in_E']=float((d@E).square().sum()/energy) if energy>0 else float('nan')
        row['loss_pcl']=float(lp.detach());row['loss_trim_unweighted']=float(lt.detach())
        del gp,gt,vp,vt,lp,lt,targets,d
    row['seconds']=time.time()-t;rows.append(row)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    write_csv(out/'grad_compare.csv',[{k:r.get(k,'') for k in keys} for r in rows]);print('X1_BATCH',row,flush=True)
    del losses,b;capture.clear()
assert len(rows)==a.batches
valid=[r for r in rows if r['triggered']]
summary=dict(model=a.model,domain=a.domain,batches=len(rows),triggered=len(valid),trigger_ratio=len(valid)/len(rows),initialized=bool(bank.initialized.item()),num_updates_start=rows[0]['num_updates_before'],num_updates_end=int(bank.num_updates.item()),trim_alpha=float(h.causal_trim_alpha),num_pcs=int(h.causal_num_pcs),seed=int(cfg.seed))
for k in ['param_cos_pcl_trim','param_norm_ratio','param_cos_total_pcl','key_cos_pcl_trim','key_norm_ratio','key_cos_total_pcl','key_diff_energy_in_E']:
    vs=[r[k] for r in valid];summary[k+'_mean']=float(np.mean(vs)) if vs else None;summary[k+'_std']=float(np.std(vs,ddof=1)) if len(vs)>1 else None
dump(out/'grad_summary.json',summary);write_csv(out/'grad_summary.csv',[summary]);print('X1_COMPLETE',summary,flush=True)
