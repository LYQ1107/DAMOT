"""Label-free identity/nuisance bases shared by offline and teacher-box fits.

Only image membership and paired views are consumed; identity/weather labels
must not be passed into this estimator. Inputs are already L2-normalized.
"""
import numpy as np

try:
    from .ics_probe_analyze import top_eig
except ImportError:
    from ics_probe_analyze import top_eig


def label_free_bases(Z, image_id, views, k_id, k_nuis, eps_ratio=0.1):
    Z = np.asarray(Z, dtype=np.float64)
    image_id = np.asarray(image_id)
    views = [np.asarray(v, dtype=np.float64) for v in views]
    if Z.ndim != 2 or len(Z) == 0 or image_id.shape != (len(Z),):
        raise ValueError('Expected nonempty [N,D] features and [N] image IDs')
    n, dim = Z.shape
    if not views or any(v.shape != Z.shape for v in views):
        raise ValueError('At least one aligned view is required')
    if not (1 <= k_id <= dim and 1 <= k_nuis <= dim and eps_ratio > 0):
        raise ValueError('Invalid dimensions or regularization')
    if any(not np.isfinite(v).all() for v in [Z] + views):
        raise ValueError('Non-finite embeddings; rows are never silently dropped')
    if any(not np.allclose(np.linalg.norm(v, axis=1), 1., atol=1e-6)
           for v in [Z] + views):
        raise ValueError('Inputs must already be L2-normalized')

    between = []
    multi_images = 0
    for im in np.unique(image_id):
        idx = np.where(image_id == im)[0]
        if len(idx) >= 2:
            between.append(Z[idx] - Z[idx].mean(0))
            multi_images += 1
    if not between:
        raise ValueError('No image contains two or more car boxes')
    Xb = np.concatenate(between, axis=0)
    Sb = Xb.T @ Xb / len(Xb)
    # Deliberately uncentered: systematic paired-view shifts are nuisance.
    residual = np.concatenate([v - Z for v in views], axis=0)
    Sw = residual.T @ residual / len(residual)
    epsilon = float(eps_ratio * np.trace(Sw) / dim)
    if not epsilon > 0:
        raise ValueError('Within-view scatter has zero trace')
    L = np.linalg.cholesky(Sw + epsilon * np.eye(dim))
    M = np.linalg.solve(L, np.linalg.solve(L, Sb).T).T
    eigvals, U = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    V = np.linalg.solve(L.T, U[:, order[:k_id]])
    E_id = np.linalg.qr(V)[0]
    E_n = top_eig(Sw, k_nuis)
    stats = dict(n_fit_rois=n, n_fit_images=int(len(np.unique(image_id))),
                 n_multi_box_images=multi_images, n_between=len(Xb),
                 n_within=len(residual), eps=epsilon, eps_ratio=float(eps_ratio),
                 within_trace=float(np.trace(Sw)),
                 top_eigs=eigvals[order[:k_id]].tolist())
    return E_id, E_n, stats
