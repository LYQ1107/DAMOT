dataset_type = 'BDDTrackingDataset'
img_prefix = '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast/'
ann_root = 'data/bdd_da/annotations/box_track_20/'
classes = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
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
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1296, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    persistent_workers=True,
    train=[
        dict(
            type='BDDTrackingDataset',
            ann_file=
            'data/bdd_da/annotations/box_track_20/box_track_dawn_dusk_all_cocofmt.json',
            img_prefix=
            '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_dawn_dusk/',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(
                num_ref_imgs=1, frame_range=3, method='uniform'),
            pipeline=[
                dict(type='LoadMultiImagesFromFile', to_float32=True),
                dict(
                    type='SeqLoadAnnotations', with_bbox=True,
                    with_track=True),
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
            classes=('pedestrian', 'car', 'truck', 'bus', 'motorcycle',
                     'bicycle'))
    ],
    val=dict(
        type='BDDTrackingDataset',
        ann_file=
        'data/bdd_da/annotations/box_track_20/box_track_overcast_all_cocofmt.json',
        img_prefix=
        '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast/',
        ref_img_sampler=None,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1296, 720),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ],
        classes=('pedestrian', 'car', 'truck', 'bus', 'motorcycle',
                 'bicycle')),
    test=dict(
        type='BDDTrackingDataset',
        ann_file=
        'data/bdd_da/annotations/box_track_20/box_track_overcast_all_cocofmt.json',
        img_prefix=
        '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast/',
        ref_img_sampler=None,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1296, 720),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ],
        classes=('pedestrian', 'car', 'truck', 'bus', 'motorcycle',
                 'bicycle')))
model = dict(
    detector=dict(
        type='FasterRCNN',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='caffe',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                clip_border=False),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=0.1111111111111111,
                loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=6,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                    clip_border=False),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            train_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100),
            pretrained=None),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
        )),
    type='DARTHQDTrack',
    track_head=dict(
        type='CausalQuasiDenseTrackHead',
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        embed_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3))),
        use_causal_direction_bank=True,
        causal_pca_mode='prototype_centered_cov_ema',
        causal_num_pcs=2,
        causal_start_pc=0,
        causal_trim_alpha=0.12,
        causal_loss_weight=0.03,
        causal_min_samples=96,
        causal_update_momentum=0.9,
        causal_normalize_feats=True,
        causal_bank_scope='global',
        causal_feat_dim=256,
        use_roi_proto_memory=True,
        roi_proto_memory_mode='stage',
        roi_proto_class_ids=(1, ),
        roi_proto_score_thr=0.75,
        roi_proto_k_per_domain_class=100,
        roi_proto_max_candidates_per_domain=1000,
        roi_proto_topk=3,
        roi_proto_loss_weight=0.005,
        roi_proto_max_rois_per_batch=64,
        roi_proto_diverse_select=False,
        roi_proto_feature_source='student',
        roi_proto_log_interval=50,
        causal_update_by_covariance=True,
        causal_cov_momentum=0.98,
        causal_qr_orthogonalize=True,
        causal_class_ids=None,
        causal_covariance_shrinkage=0.0,
        causal_proto_center_momentum=0.95,
        causal_proto_center_min_count=8),
    tracker=dict(
        type='QDTracker',
        init_score_thr=0.5,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'),
    init_cfg=dict(
        type='KDPretrained',
        checkpoint=
        'https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth',
        map_location='cpu'),
    teacher=dict(eval_teacher_preds=False, eval_mode_teacher=True),
    loss_rpn_distillation=dict(
        type='RPNDistillationLoss', loss_weight=0.1, reg_valid_threshold=0.1),
    loss_roi_distillation=dict(type='ROIDistillationLoss', loss_weight=0.1),
    teacher_pipeline=[
        dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
        dict(
            type='SeqResize',
            img_scale=(1088, 1088),
            share_params=True,
            ratio_range=(0.8, 1.2),
            keep_ratio=True,
            bbox_clip_border=False),
        dict(
            type='SeqRandomCrop',
            share_params=False,
            crop_size=(1088, 1088),
            bbox_clip_border=False,
            allow_negative_crop=True)
    ],
    student_pipeline=[
        dict(type='SeqPhotoMetricDistortion', share_params=True),
        dict(
            type='SeqSPAFrequencyTransform',
            mode='low_freq_mask',
            low_ratio=0.1,
            keep_prob=0.9,
            apply_noise=False,
            noise_ratio=0.0)
    ],
    contrastive_pipeline=[
        dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
        dict(
            type='SeqResize',
            img_scale=(1088, 1088),
            share_params=True,
            ratio_range=(0.8, 1.2),
            keep_ratio=True,
            bbox_clip_border=False),
        dict(
            type='SeqRandomCrop',
            share_params=False,
            crop_size=(1088, 1088),
            bbox_clip_border=False,
            allow_negative_crop=True),
        dict(type='SeqPhotoMetricDistortion', share_params=True)
    ],
    out_pipeline=[
        dict(
            type='SeqNormalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='SeqPad', size_divisor=32),
        dict(type='MatchInstances', skip_nomatch=False),
        dict(
            type='VideoCollect',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_scores',
                'gt_match_indices', 'gt_instance_ids'
            ])
    ],
    conf_thr=0.7,
    use_spa_frequency_aug=False,
    log_view_stats=True)
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    type='CustomOptimizerHook', grad_clip=dict(max_norm=35.0, norm_type=2))
checkpoint_config = dict(interval=1, max_keep_ckpts=-1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
lr_config = dict(
    policy='step',
    step=[8, 11],
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)
total_epochs = 5
evaluation = dict(
    metric=['bbox', 'track'],
    interval=6,
    results_kwargs=dict(
        results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(
        iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_by_classes=False,
        nproc=1,
        majority_voting=False))
teacher_pipeline = [
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        bbox_clip_border=False,
        allow_negative_crop=True)
]
student_pipeline = [dict(type='SeqPhotoMetricDistortion', share_params=True)]
contrastive_pipeline = [
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        bbox_clip_border=False,
        allow_negative_crop=True),
    dict(type='SeqPhotoMetricDistortion', share_params=True)
]
norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
out_pipeline = [
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=False),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ])
]
custom_hooks = [
    dict(type='EMATrainHook', momentum=0.002, interval=1, warm_up=100)
]
spa_student_pipeline = [
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqSPAFrequencyTransform',
        mode='low_freq_mask',
        low_ratio=0.1,
        keep_prob=0.9,
        apply_noise=False,
        noise_ratio=0.0)
]
seed = 388086182
gpu_ids = range(0, 4)
work_dir = 'work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoCenter_roiV1Weak_4domain_5e/episode/adapt/stage03_after_dawn_dusk'
