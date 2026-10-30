"""Inference-only raw-embedding shrinkage; the default path is exact identity."""
import numpy as np
import torch
from mmdet.models import HEADS
from mmtrack.models.track_heads import QuasiDenseTrackHead


@HEADS.register_module(force=True)
class ICSInferenceTrackHead(QuasiDenseTrackHead):
    def __init__(self,inference_trim_basis=None,inference_trim_beta=None,
                 inference_trim_enabled=None,basis_source=None,**kwargs):
        self.default_trim_alpha=float(kwargs.get('causal_trim_alpha',.25))
        super().__init__(**kwargs)
        self.inference_trim_basis_path=inference_trim_basis
        self.basis_source=basis_source or ('npy' if inference_trim_basis else 'checkpoint_bank')
        self.inference_trim_enabled=(inference_trim_beta is not None and float(inference_trim_beta)!=0.) if inference_trim_enabled is None else bool(inference_trim_enabled)
        self.inference_trim_beta=self.default_trim_alpha if inference_trim_beta is None else float(inference_trim_beta)
        self.register_buffer('_checkpoint_pcs',torch.empty(0),persistent=False)
        self.register_buffer('_inference_trim_basis',torch.empty(0),persistent=False)
        self._basis_cache={};self.inference_trim_calls=0;self.inference_trim_min_rois=None

    def load_checkpoint_bank(self,pcs,config_alpha=None):
        """The loader passes the STUDENT track_head.causal_bank.pcs verbatim."""
        if pcs is not None:self._checkpoint_pcs=pcs.detach().clone()
        if config_alpha is not None:self.default_trim_alpha=float(config_alpha)
        self._basis_cache.clear()

    def set_inference_intervention(self,beta=None,basis_source=None,npy_path=None,enabled=True):
        self.inference_trim_beta=self.default_trim_alpha if beta is None else float(beta)
        self.inference_trim_enabled=bool(enabled)
        if basis_source is not None:self.basis_source=basis_source
        if npy_path is not None:self.inference_trim_basis_path=str(npy_path)

    def orthogonal_basis(self,z):
        source=self.basis_source;key=(source,self.inference_trim_basis_path,str(z.device),str(z.dtype))
        if key in self._basis_cache:return self._basis_cache[key]
        if source=='checkpoint_bank':
            if self._checkpoint_pcs.numel()==0:raise RuntimeError('Checkpoint has no student causal_bank.pcs')
            # Checkpoint PCs are ROWS. Do not substitute covariance eigenvectors.
            columns=self._checkpoint_pcs.detach().cpu().double().numpy().T
        elif source=='npy':
            matrix=np.asarray(np.load(self.inference_trim_basis_path),dtype=np.float64)
            if matrix.ndim==1:matrix=matrix[:,None]
            if matrix.shape[0]==256:columns=matrix
            elif matrix.shape[1]==256:columns=matrix.T
            else:raise ValueError('Basis must have feature dimension256')
        else:raise ValueError('Unknown basis_source: '+str(source))
        if columns.ndim!=2 or columns.shape[0]!=256 or not np.isfinite(columns).all():raise ValueError('Invalid basis')
        q,r=np.linalg.qr(columns,mode='reduced')
        if np.any(np.abs(np.diag(r))<1e-10):raise ValueError('Rank-deficient basis')
        E=torch.from_numpy(np.ascontiguousarray(q)).to(device=z.device,dtype=z.dtype)
        self._inference_trim_basis=E;self._basis_cache[key]=E
        return E

    def trim_raw_embedding(self,z):
        # No normalization and no RoI-count gate. Never call bank.trim().
        if self.training or not self.inference_trim_enabled or self.inference_trim_beta==0. or z is None or z.numel()==0:return z
        E=self.orthogonal_basis(z)
        self.inference_trim_calls+=1
        count=int(z.shape[0]);self.inference_trim_min_rois=count if self.inference_trim_min_rois is None else min(count,self.inference_trim_min_rois)
        return z-self.inference_trim_beta*(z@E)@E.T

    def extract_bbox_feats(self,x,bboxes):
        return self.trim_raw_embedding(super().extract_bbox_feats(x,bboxes))
