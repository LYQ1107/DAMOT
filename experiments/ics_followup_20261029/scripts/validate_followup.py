"""Protocol and algebra checks on actual outputs, including raw Fisher."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'code/tools/analysis'),str(ROOT/'code/tools/run')]
from apr_complement_experiment import torch,np,Config
import ics_probe_analyze as p
from darth.models.track_heads.inference_subspace_track_head import project_association_features
from protocol import write_json,write_views
checks={}
x=np.array([[1.,1.],[3.,1.],[-1.,1.],[-3.,1.]])
weather=np.array(['a','a','b','b']);cls=np.ones(4)
assert abs(p.fisher_ratio(x,weather,cls)-4)<1e-9
assert abs(p.fisher_ratio(7*x,weather,cls)-4)<1e-9
checks['Fisher_raw_known_value_and_scale_invariance']=True
rank=p.rank_basis(np.array([[1.,0.],[2.,0.]]),16)
assert rank.shape==(2,1);checks['Oracle_rank_truncation']=True
t=torch.randn(30,256);E=torch.linalg.qr(torch.randn(256,16))[0].T.contiguous()
assert project_association_features(t,E,0.) is t
out=project_association_features(t,E,1.)
assert (out@E.T).abs().max().item()<3e-6
assert not torch.allclose(out.norm(dim=1),torch.ones(30))
checks['S4_beta0_exact_identity_and_raw_beta1_projection']=True
result=ROOT/'results/analysis_F_stage4_e5_v2'
rows=list(csv.DictReader((result/'csv/P3_intervention.csv').open()))
assert len(rows)==78
for r in rows:
    if r['subspace']=='Random':assert r['random_repeats']=='10'
    if r['subspace']=='Weather oracle':assert int(r['k_effective'])<=4
    if float(r['beta'])==0:
        assert abs(float(r['id_mAP'])-.5731398500607996)<1e-12
checks['S2_complete_78_cell_grid_and_frozen_beta0']=True
original=Path('/data1/liuyeqiang/damot_rebuttal_20261026/results/ics_probe/analysis_F_stage4_e5/csv/P1_energy.csv')
assert list(csv.DictReader(original.open()))==list(csv.DictReader((result/'csv/P1_energy.csv').open()))
checks['S2_P1_matches_published_v1_exactly']=True
for r in list(csv.DictReader((ROOT/'results/drift_decomposition/F/drift_energy.csv').open())):
    assert abs(float(r['mean_in_sq'])+float(r['mean_out_sq'])-float(r['mean_total_sq']))<1e-12
checks['S3_orthogonal_energy_conservation']=True
cfg=Config.fromfile('/data1/liuyeqiang/damot_rebuttal_20261026/configs/F_stage4_e5_embedded.py')
for dest in [result,ROOT/'results/drift_decomposition/F']:
    write_views(cfg,dest/'resolved_views.txt',dict(feature_transform='fixed geometry, normalized clean embeddings',inference_training_views_executed=False))
write_json(result/'config_diff.json',dict(model_changes=[],analysis_version=2,seed=0,k=[2,4,8,16,32],betas=[0,.5,1],random_repeats=10,Fisher='raw unrenormalized projection on car/test',stored_vs_covariance='separate per user approval; equivalence check failed',original_v1_preserved=True))
write_json(ROOT/'manifests/numerical_validation.json',dict(status='PASS',checks=checks))
print(json.dumps(checks,indent=2))
