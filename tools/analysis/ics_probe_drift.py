"""S3: fixed-reference drift and cross-stage retrieval; CPU only."""
import argparse,json,os
from pathlib import Path
import numpy as np
import ics_probe_analyze as p
DOMAINS=['clear_holdout_45v','overcast','rainy','dawn_dusk','night']
TARGETS=DOMAINS[1:]

def load(root,label,fap=False):
    parts=[];bank=None
    keys=['video','instance_id','frame_id','image_id','bbox','category','weather','feat_clean']+(['feat_fap'] if fap else [])
    for d in DOMAINS:
        with np.load(str(root/label/(d+'.npz')),allow_pickle=False) as z:
            part={k:z[k] for k in keys};parts.append(part)
            if bank is None:bank={k[5:]:z[k] for k in z.files if k.startswith('bank_')}
    data={k:np.concatenate([v[k] for v in parts]) for k in keys}
    data['Z']=p.normalize(data.pop('feat_clean').astype(float))
    return data,bank

def cross_map(query,gallery,tasks):
    q,g=p.normalize(query),p.normalize(gallery)
    return float(np.mean([p.average_precision(g[gg]@q[qq],pos) for qq,gg,pos in tasks]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--feat-root',type=Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'complete.json').exists():return
    ref,_=load(a.feat_root,'F_stage1_e5',True);fit=p.split_mask(ref['video'],0);te=~fit
    z=ref['Z'];res=p.normalize(ref['feat_fap'].astype(float))-z
    iid=p.identity_rows(z[fit],ref['video'][fit],ref['instance_id'][fit],ref['category'][fit])
    ww=p.between_weather_rows(z[fit],ref['weather'][fit],ref['category'][fit])
    B={}
    for m in [2,8,16]:
        B[('FAP residual PCA',m)]=p.rank_basis(res[fit],m,True)
        B[('Identity oracle',m)]=p.rank_basis(iid,m)
        B[('Weather oracle',m)]=p.rank_basis(ww,m)
    stages={};banks={}
    for k in range(1,5):
        d,b=load(a.feat_root,'%s_stage%d_e5'%(a.model,k));stages[k]=d['Z'];banks[k]=b
        for key in ['video','instance_id','frame_id','image_id','category','bbox']:
            assert np.array_equal(ref[key],d[key]),(a.model,k,key)
    rng=np.random.RandomState(9973);random={m:[p.orth(rng.randn(256,m)) for _ in range(10)] for m in [2,16]}
    drift=[]
    for before,after in [(1,2),(2,3),(3,4),(1,4)]:
        delta=stages[after]-stages[before]
        for bucket,domains in [('old',TARGETS[:after-1]),('current',[TARGETS[after-1]])]:
            mask=te & np.isin(ref['weather'],domains);D=delta[mask];total=(D*D).sum(1);valid=total>1e-20
            for m in [2,16]:
                options={name:[B[(name,m)]] for name in ['FAP residual PCA','Weather oracle','Identity oracle']}
                options['ICS stored bank_pcs' if m==2 else 'ICS covariance eigenvectors']=[p.orth(banks[after]['pcs'].T.astype(float)) if m==2 else p.top_eig(banks[after]['cov_bank'].astype(float),m)]
                options['Random']=random[m]
                for name,bases in options.items():
                    vals=[((D@E)**2).sum(1) for E in bases];shares=[float(np.mean(v[valid]/total[valid])) for v in vals]
                    drift.append(dict(model=a.model,from_stage=before,to_stage=after,bucket=bucket,domains=','.join(domains),basis=name,k_requested=m,k_effective=bases[0].shape[1],rois=int(mask.sum()),mean_energy_share=float(np.mean(shares)),random_std=float(np.std(shares)),mean_in_sq=float(np.mean(vals)),mean_total_sq=float(total.mean()),mean_out_sq=float(total.mean()-np.mean(vals)),ratio_of_sums=float(np.mean([v.sum()/total.sum() for v in vals])),basis_reference='F_stage1_e5 fit videos; ICS from destination checkpoint',split='test',seed=0))
    p.write_csv(str(a.out/'drift_energy.csv'),drift)
    retrieval=[];restoration=[]
    for k in range(2,5):
        for j in range(1,k):
            domain=TARGETS[j-1];mask=te & (ref['weather']==domain)
            tasks=p.build_retrieval(ref['video'][mask],ref['frame_id'][mask],ref['instance_id'][mask],4,30,4000,0)
            old,new=stages[j][mask],stages[k][mask];same=cross_map(old,old,tasks);cross=cross_map(new,old,tasks)
            row=dict(model=a.model,query_stage=k,gallery_stage=j,domain=domain,queries=len(tasks),mAP_same=same,mAP_cross=cross,forgetting_proxy=same-cross,seed=0,gap=4,window=30)
            retrieval.append(row)
            for name in ['FAP residual PCA','Identity oracle']:
                for m in [2,8,16]:
                    E=B[(name,m)];fixed=new-((new-old)@E)@E.T;restored=cross_map(fixed,old,tasks)
                    restoration.append(dict(row,basis=name,k_requested=m,k_effective=E.shape[1],mAP_restored=restored,forgetting_restored=same-restored,forgetting_reduction=restored-cross))
    p.write_csv(str(a.out/'cross_stage_retrieval.csv'),retrieval);p.write_csv(str(a.out/'drift_restoration.csv'),restoration)
    protocol=dict(model=a.model,seed=0,basis_reference='F_stage1_e5 (literal requested common reference for F/HF/NOGEOM)',fit_videos=np.unique(ref['video'][fit]).tolist(),statistics_split='test',cross_model_coordinate_alignment='none; output coordinates shared by convention, not proven gauge-invariant',config_changes=[],ICS_definition='stored PCS at k2, covariance eigenvectors at k16 (non-equivalence disclosed)',same_roi_metadata=True)
    (a.out/'config_diff.json').write_text(json.dumps(protocol,indent=2))
    (a.out/'complete.json').write_text(json.dumps(dict(status='complete',drift_rows=len(drift),retrieval_rows=len(retrieval),restoration_rows=len(restoration)),indent=2))
    print('S3 COMPLETE',a.model,flush=True)

if __name__=='__main__':main()
