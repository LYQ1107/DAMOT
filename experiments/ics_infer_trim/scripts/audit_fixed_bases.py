"""Verify copied C2/C4 fixed bases equal their actual checkpoint bank spans."""
from common import *
from bank_audit import audit
from scipy.linalg import subspace_angles
records=[]
for label in ['C2','C4']:
    record=audit(checkpoint(label),label)
    file=ROOT/'inputs'/(label+'_basis.npy');fixed=np.asarray(np.load(str(file)),dtype=float)
    if fixed.shape[0]!=256:fixed=fixed.T
    q=np.linalg.qr(fixed)[0];stored=np.load(str(ROOT/'inputs/bank_audit'/(label+'_QR_columns.npy')))
    angles=np.degrees(subspace_angles(q,stored))
    assert max(angles)<.001,(label,angles)
    records.append(dict(model=label,basis_file=str(file),basis_sha256=sha(file),checkpoint_sha256=record['checkpoint_sha256'],
                        max_principal_angle_deg=float(max(angles)),training_alpha=record['training_alpha'],mode=record['causal_pca_mode']))
dump(ROOT/'manifests/fixed_basis_provenance.json',dict(status='PASS',models=records))
print('FIXED_BASES_MATCH_CHECKPOINTS',flush=True)
