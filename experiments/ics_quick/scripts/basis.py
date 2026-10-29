"""X2 basis: per-class centered, normalized clean car embeddings."""
from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--model',choices=['F','HF'],required=True);a=p.parse_args();setup(a.model)
import torch,numpy as np
path=Path(INPUTS['F_features']) if a.model=='F' else Q/'feats/HF_stage4_e5/clear_holdout_45v.npz'
with np.load(str(path),allow_pickle=True) as z:
    cat=z['category'];x=z['feat_clean'].astype(np.float64)
    print('categories',np.unique(cat),flush=True)
    mask=(cat=='car') if cat.dtype.kind in 'OUS' else (cat==1)
    x=x[mask];x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);x-=x.mean(axis=0,keepdims=True)
    vals,vecs=np.linalg.eigh(x.T@x/(len(x)-1));E=vecs[:,np.argsort(vals)[-2:][::-1]]
ck=torch.load(str(checkpoint(a.model)),map_location='cpu');pcs=ck['state_dict']['track_head.causal_bank.pcs'].numpy().T
U=np.linalg.qr(pcs)[0];angles=np.rad2deg(np.arccos(np.clip(np.linalg.svd(E.T@U,compute_uv=False),0,1)))
np.save(str(Q/'X2'/('basis_e_%s.npy'%a.model)),E.astype(np.float32))
dump(Q/'X2'/('basis_e_%s.json'%a.model),dict(model=a.model,features=str(path),feature_sha256=sha(path),N=len(x),classes='car',l2_normalize_before_class_centering=True,principal_angles_degrees=angles.tolist(),explained_variance_top2=(np.sort(vals)[-2:][::-1]/vals.sum()).tolist(),basis_sha256=sha(Q/'X2'/('basis_e_%s.npy'%a.model))))
print('X2_BASIS',a.model,angles,flush=True)
