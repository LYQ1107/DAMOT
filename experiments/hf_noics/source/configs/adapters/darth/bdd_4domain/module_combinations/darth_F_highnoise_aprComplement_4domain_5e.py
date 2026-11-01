"""F + contrastive noise + complement-only APR; no ICS estimator change."""
_base_ = ['./darth_spaP1_causalProtoShrink_roiV1Weak_4domain_5e.py']

model = dict(
    track_head=dict(
        roi_proto_couple_ics=True,
        roi_proto_couple_beta=1.0,
        roi_proto_couple_metric='projected_cosine'),
    contrastive_pipeline=[
        dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
        dict(type='SeqResize', img_scale=(1088, 1088), share_params=True,
             ratio_range=(0.8, 1.2), keep_ratio=True, bbox_clip_border=False),
        dict(type='SeqRandomCrop', share_params=False, crop_size=(1088, 1088),
             bbox_clip_border=False, allow_negative_crop=True),
        dict(type='SeqPhotoMetricDistortion', share_params=True),
        # Historical "high_freq_noise" is spatial Gaussian noise mixing,
        # not an explicit high-pass Fourier filter. Keep its existing semantics.
        dict(type='SeqSPAFrequencyTransform', mode='high_freq_noise',
             apply_noise=True, noise_ratio=0.03)])
