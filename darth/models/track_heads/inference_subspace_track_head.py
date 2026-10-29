"""Optional association-only inference intervention; disabled by default."""
import numpy as np
import torch
from mmdet.models.builder import HEADS
from mmtrack.models.track_heads import QuasiDenseTrackHead

def project_association_features(features,basis,beta):
    if beta == 0 or basis is None:
        return features
    E=basis.to(device=features.device,dtype=features.dtype)
    return features-beta*(features@E.t())@E

@HEADS.register_module()
class SubspaceInterventionQuasiDenseTrackHead(QuasiDenseTrackHead):
    def __init__(self,*args,subspace_intervention=None,**kwargs):
        super().__init__(*args,**kwargs)
        cfg=subspace_intervention or {}
        self.intervention_enabled=bool(cfg.get('enabled',False))
        self.intervention_beta=float(cfg.get('beta',0.))
        basis=np.load(cfg['basis_file'],allow_pickle=False) if cfg.get('basis_file') else np.zeros((0,256),np.float32)
        if len(basis):
            assert basis.ndim==2 and basis.shape[1]==256 and np.isfinite(basis).all()
            assert np.allclose(basis@basis.T,np.eye(len(basis)),atol=1e-5)
        self.register_buffer('_intervention_basis',torch.from_numpy(basis.astype(np.float32)),persistent=False)

    def extract_bbox_feats(self,x,bboxes):
        features=super().extract_bbox_feats(x,bboxes)
        if not self.training and self.intervention_enabled:
            assert self._intervention_basis.numel()>0
            return project_association_features(features,self._intervention_basis,self.intervention_beta)
        return features
