"""Check actual head behavior, including single-RoI and exact disabled paths."""
from common import *
import torch
from mmcv import Config
from darth.models.track_heads.ics_inference_track_head import ICSInferenceTrackHead
cfg=Config.fromfile(str(CROOT/'evaluation/C2/night/used_config.py'))
spec=dict(cfg.model.track_head);spec.pop('type');spec.pop('inference_trim_beta',None);spec.pop('inference_trim_basis',None)
h=ICSInferenceTrackHead(**spec).eval();z=torch.randn(1,256)*13
checks=[]
assert not h.inference_trim_enabled and h.trim_raw_embedding(z) is z;checks.append('default disabled is exact identity')
h.set_inference_intervention(beta=0);assert h.trim_raw_embedding(z) is z;checks.append('beta0 returns same tensor object')
h.set_inference_intervention(beta=.25,basis_source='npy',npy_path=ROOT/'inputs/C2_basis.npy')
E=h.orthogonal_basis(z);got=h.trim_raw_embedding(z);expected=z-.25*(z@E)@E.T
assert torch.equal(got,expected) and not torch.equal(got,z);checks.append('one RoI shrinks by raw matrix operator without min_samples gate')
assert torch.allclose(got.norm(),expected.norm()) and abs(float(got.norm())-1)>1;checks.append('no feature normalization')
h.train();assert h.trim_raw_embedding(z) is z;h.eval();checks.append('training path unchanged')
pcs=torch.randn(2,256);pcs[1]+=.3*pcs[0];h.load_checkpoint_bank(pcs,config_alpha=.1)
h.set_inference_intervention(beta=None,basis_source='checkpoint_bank');q=h.orthogonal_basis(z)
assert h.inference_trim_beta==.1 and torch.allclose(q.T@q,torch.eye(2),atol=1e-6)
assert torch.equal(h.trim_raw_embedding(z),z-.1*(z@q)@q.T);checks.append('checkpoint row PCs QR and actual config alpha default')
empty=torch.empty(0,256);assert h.trim_raw_embedding(empty) is empty;checks.append('empty embedding exact identity')
dump(ROOT/'manifests/selftest.json',dict(status='PASS',checks=checks))
print('SELFTEST_PASS',len(checks),flush=True)
