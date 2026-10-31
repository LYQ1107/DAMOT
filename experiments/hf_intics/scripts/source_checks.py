"""Check untouched APR/trim code and the specified covariance math on CPU."""
import ast
import importlib.util
import json
from pathlib import Path
import torch
Q=Path(__file__).resolve().parents[1]

def method(root,relative,name):
    tree=ast.parse((root/relative).read_text())
    found=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    assert len(found)==1
    return ast.dump(found[0],include_attributes=False)

head='darth/models/track_heads/causal_quasi_dense_track_head.py'
bank='darth/models/utils/roi_causal_bank.py'
for relative,name in [(head,'_get_apr_basis'),(bank,'trim')]:
    assert method(Q/'code',relative,name)==method(Q/'code_original',relative,name)

def load(root,label):
    spec=importlib.util.spec_from_file_location(label,str(root/bank))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.CausalDirectionBank

Old=load(Q/'code_original','old_bank')
New=load(Q/'code','new_bank')
X=(torch.arange(64*256,dtype=torch.float32).reshape(64,256)/10000)+.2
old=Old(256,2,min_samples=32,pca_mode='cov_ema')
default=New(256,2,min_samples=32,pca_mode='cov_ema')
assert torch.equal(old.compute_covariance(X),default.compute_covariance(X))
new=New(256,2,min_samples=32,pca_mode='intervention_residual',
    normalize=False,center=False,cov_momentum=.98,covariance_shrinkage=0)
assert torch.equal(new.compute_covariance(X),X.T @ X / 63)
assert new.update(X)
orth_error=float((new.pcs @ new.pcs.T-torch.eye(2)).abs().max())
assert orth_error<1e-5
features=X.clone().requires_grad_()
new.trim(features,alpha=.25).square().sum().backward()
assert torch.isfinite(features.grad).all()
report=dict(status='passed',APR_basis_reader_AST_unchanged=True,
    trim_operator_AST_unchanged=True,default_covariance_bitwise_equal=True,
    intervention_uncentered_covariance_exact=True,residual_not_renormalized=True,
    QR_orthogonality_error=orth_error,trim_gradient_finite=True)
(Q/'preflight/source_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
