import hashlib
import numpy as np

def split_mask(groups, seed=0):
    """Deterministic 50/50 split on group names (video or scene)."""
    u = np.unique(groups)
    fit = set(g for g in u if int(hashlib.md5(('%s/%d' % (g, seed)).encode())
                                  .hexdigest()[:8], 16) % 2 == 0)
    if len(fit) in (0, len(u)):           # degenerate: alternate
        fit = set(u[::2])
    return np.array([g in fit for g in groups])
