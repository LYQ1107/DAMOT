# ICS subspace probe, control run (R2): identical to the final DAMOT config
# (spaP1_causalProtoShrink_roiV1Weak) except that the ICS bank is estimated
# from PAIRED residuals (student/FAP view minus contrastive view of the same
# pseudo box) instead of class-centred features. Used only to compare which
# estimator yields a weather-dominated subspace.
_base_ = [
    './darth_spaP1_causalProtoShrink_roiV1Weak_4domain_5e.py',
]

model = dict(
    track_head=dict(
        causal_pca_mode='pairwise_residual',
    ),
)
