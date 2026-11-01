# Reference patch objective: shrinkage distance on normalized features.
# This shares the ICS subspace; original ICS trim itself is unchanged.
_base_ = ['./darth_spaP1_causalProtoShrink_roiV1Weak_4domain_5e.py']
model = dict(track_head=dict(
    roi_proto_couple_ics=True, roi_proto_couple_beta=None,
    roi_proto_couple_basis='ics', roi_proto_couple_metric='projected_l2'))
