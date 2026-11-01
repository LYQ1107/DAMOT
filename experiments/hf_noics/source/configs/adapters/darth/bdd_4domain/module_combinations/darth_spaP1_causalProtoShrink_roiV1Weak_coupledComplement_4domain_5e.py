_base_ = ['./darth_spaP1_causalProtoShrink_roiV1Weak_coupled_4domain_5e.py']
model = dict(track_head=dict(roi_proto_couple_beta=1.0))
