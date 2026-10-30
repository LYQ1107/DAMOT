"""Optional inference-only linear embedding intervention; default is identity."""
import numpy as np
import torch
from mmdet.models import HEADS
from mmtrack.models.track_heads import QuasiDenseTrackHead

@HEADS.register_module()
class ICSInferenceTrackHead(QuasiDenseTrackHead):
    def __init__(self, inference_trim_basis=None, inference_trim_beta=0., **kwargs):
        super().__init__(**kwargs)
        self.inference_trim_beta=float(inference_trim_beta)
        self.inference_trim_basis_path=inference_trim_basis
        E=torch.empty(0)
        if inference_trim_basis is not None:
            E=torch.from_numpy(np.load(inference_trim_basis)).float()
            assert E.ndim==2 and E.shape[0]==256
            assert torch.allclose(E.T@E,torch.eye(E.shape[1]),atol=1e-5,rtol=1e-5)
        self.register_buffer('_inference_trim_basis',E,persistent=False)

    def extract_bbox_feats(self,x,bboxes):
        z=super().extract_bbox_feats(x,bboxes)
        if not self.training and self.inference_trim_beta!=0.:
            E=self._inference_trim_basis.to(z)
            assert E.numel()>0
            return z-self.inference_trim_beta*(z@E)@E.T
        return z
