"""Export actual stored bank PCs and measure requested vs sequential operators."""
import argparse
from common import *
def audit(checkpoint_path,label):
    import torch
    from mmcv import Config
    ck=torch.load(str(checkpoint_path),map_location='cpu');state=ck['state_dict']
    pcs=state.get('track_head.causal_bank.pcs')
    if pcs is None:raise RuntimeError('Missing student checkpoint causal_bank.pcs')
    config=Config.fromstring(ck['meta']['config'],'.py').model.track_head
    alpha=float(config.causal_trim_alpha)
    rows=pcs.detach().cpu().double().numpy();assert rows.ndim==2 and rows.shape[1]==256
    q,r=np.linalg.qr(rows.T);assert np.all(np.abs(np.diag(r))>1e-10)
    root=ROOT/'inputs/bank_audit';root.mkdir(parents=True,exist_ok=True)
    np.save(str(root/(label+'_stored_row_pcs.npy')),rows);np.save(str(root/(label+'_QR_columns.npy')),q)
    eye=np.eye(256);sequential=eye.copy()
    for pc in rows:sequential=sequential@(eye-alpha*np.outer(pc,pc))
    requested=eye-alpha*q@q.T;delta=np.linalg.norm(sequential-requested,'fro')
    record=dict(checkpoint=str(checkpoint_path),checkpoint_sha256=sha(checkpoint_path),pcs_key='track_head.causal_bank.pcs',
         rows=rows.shape[0],dimension=256,training_alpha=alpha,causal_pca_mode=config.get('causal_pca_mode'),
         causal_fixed_basis_path=config.get('causal_fixed_basis_path',config.get('causal_fixed_pcs_path')),
         row_gram=(rows@rows.T).tolist(),QR_orthogonality_error=float(np.linalg.norm(q.T@q-np.eye(len(rows)))),
         sequential_training_vs_QR_operator_frobenius=float(delta),
         relative_to_training_removed_operator=float(delta/max(np.linalg.norm(eye-sequential,'fro'),1e-30)),
         note='Requested QR subspace projection; sequential training trim is identical only for orthonormal stored PCs (up to rounding).')
    dump(root/(label+'.json'),record);print('BANK_AUDIT',label,record['relative_to_training_removed_operator'],flush=True)
    return record
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--label',required=True);a=p.parse_args();audit(Path(a.checkpoint),a.label)
