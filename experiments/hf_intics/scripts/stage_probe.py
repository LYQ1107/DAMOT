"""Frozen stage1 gate / final P1 features, original canonical P1 sampling."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
Q=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Q/'probe_tools'))
import ics_probe_analyze as p
DOMAINS=['clear_holdout_45v','overcast','rainy','dawn_dusk','night']
STAGES=['overcast','rainy','dawn_dusk','night']
arg=argparse.ArgumentParser()
arg.add_argument('--stage',type=int,required=True)
arg.add_argument('--gpu',required=True)
a=arg.parse_args()
stage=a.stage
root=Q/'probe'/('stage%d'%stage)
label='INTICS_stage%d_e5'%stage
checkpoint=Q/'episode/stage_snapshots'/('stage%02d_after_%s.pth'%(stage,STAGES[stage-1]))
root.mkdir(parents=True,exist_ok=True)
command=['/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python',
    str(Q/'code/tools/analysis/ics_probe_extract.py'),'--checkpoint',str(checkpoint),
    '--config',str(Q/'configs'/('stage%d.py'%stage)),'--label',label,'--out-dir',str(root/'features'),
    '--views','clean,fap' if stage==1 else 'clean,fap,photo,student',
    '--dose-frames-per-video','0' if stage==1 else '3']
for domain in DOMAINS:
    command+=['--domain',domain,'/data1/liuyeqiang/damot_rebuttal_20261026/data/annotations/'+domain+'.json',str(Q/'probe_data'/domain)]
(root/'extract_command.json').write_text(json.dumps(dict(command=command,gpu=a.gpu,
    checkpoint=str(checkpoint),canonical_P1_defaults_preserved=True),indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU',PYTHONDONTWRITEBYTECODE='1')
if not (root/'features'/label/'meta.json').exists():
    with (root/'extract.log').open('w') as log:
        subprocess.run(command,cwd=str(Q/'code'),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
data,bank=p.load_label(str(root/'features'),label,DOMAINS,[1])
assert set(map(str,np.unique(data['weather'])))==set(DOMAINS)
Z=p.normalize(data['feat_clean'].astype(float))
fit=p.split_mask(data['video'],0)
residual=p.normalize(data['feat_fap'].astype(float))-Z
offline=p.top_eig(residual[fit].T @ residual[fit]/fit.sum(),2)
pcs=bank['pcs'].astype(float)
assert pcs.shape==(2,256)
orthogonality_error=float(np.abs(pcs @ pcs.T-np.eye(2)).max())
assert orthogonality_error<1e-5
E=p.orth(pcs.T)
first=pcs[0]/np.linalg.norm(pcs[0])
angle=float(np.degrees(np.arccos(np.clip(abs(first @ offline[:,0]),-1.,1.))))
to_span=float(np.degrees(np.arccos(np.clip(np.linalg.norm(first @ offline),-1.,1.))))
angles=np.degrees(np.arccos(np.clip(np.linalg.svd(E.T @ offline,compute_uv=False),-1.,1.)))
test=~fit
W=np.concatenate([p.shift_rows(Z[test],data['weather'][test],data['category'][test],
    domain,'clear_holdout_45v') for domain in DOMAINS if domain!='clear_holdout_45v'],axis=0)
ID=p.identity_rows(Z[test],data['video'][test],data['instance_id'][test],data['category'][test])
report=dict(stage=stage,split_seed=0,fit_rois=int(fit.sum()),test_rois=int(test.sum()),
    bank_shape=list(pcs.shape),bank_orthogonality_error=orthogonality_error,
    offline_basis='uncentered FAP-minus-clean, normalized embeddings, fit videos only',
    gate_angle_definition='principal angle bank PC1 to offline intervention two-dimensional column space',
    gate_angle_field='first_direction_to_offline_span_deg',gate_angle_deg=to_span,
    first_direction_angle_deg=angle,first_direction_to_offline_span_deg=to_span,
    principal_angles_deg=angles.tolist(),
    P1_reference_domain='clear_holdout_45v',
    P1_weather_shift_energy_share=p.energy(W,E),P1_identity_energy_share=p.energy(ID,E),
    P1_per_direction=[dict(bank_row=i+1,
        weather_shift_energy_share=p.energy(W,pcs[i:i+1].T/np.linalg.norm(pcs[i])),
        identity_energy_share=p.energy(ID,pcs[i:i+1].T/np.linalg.norm(pcs[i])))
        for i in range(len(pcs))],
    fit_videos=np.unique(data['video'][fit]).tolist(),test_videos=np.unique(data['video'][test]).tolist())
np.save(str(root/'offline_fap_k2.npy'),offline)
np.save(str(root/'bank_pcs.npy'),bank['pcs'])
report['offline_basis_sha256']=hashlib.sha256((root/'offline_fap_k2.npy').read_bytes()).hexdigest()
report['bank_pcs_sha256']=hashlib.sha256((root/'bank_pcs.npy').read_bytes()).hexdigest()
(root/'P1.json').write_text(json.dumps(report,indent=2)+'\n')
with (root/'P1.csv').open('w',newline='') as f:
    fields=['stage','first_direction_angle_deg','first_direction_to_offline_span_deg',
        'P1_weather_shift_energy_share','P1_identity_energy_share']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerow({k:report[k] for k in fields})
if stage==4:
    import evidence_cpu as evidence
    evidence.ROOT=root/'evidence'
    evidence.INPUTS['INTICS']=root/'features'/label
    evidence.run('INTICS')
print('STAGE_PROBE_COMPLETE',stage,angle,flush=True)
