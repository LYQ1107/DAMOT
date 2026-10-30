data = dict(train=[
    dict(
        type='BDDTrackingDataset',
        ann_file=
        '/data1/liuyeqiang/damot_rebuttal_20261026/data/annotations/night.json',
        img_prefix='/data1/liuyeqiang/ics_v2_20261030_184026/data_views/night/',
        key_img_sampler=dict(interval=1),
        ref_img_sampler=dict(num_ref_imgs=1, frame_range=3, method='uniform'),
        pipeline=[
            dict(type='LoadMultiImagesFromFile', to_float32=True),
            dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
            dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
            dict(
                type='SeqResize',
                img_scale=(1296, 720),
                share_params=True,
                keep_ratio=True,
                bbox_clip_border=False),
            dict(type='MatchInstances', skip_nomatch=True),
            dict(
                type='VideoCollect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
                    'gt_instance_ids'
                ]),
            dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
        ],
        classes=('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'))
])
