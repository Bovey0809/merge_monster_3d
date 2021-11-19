_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py',
]

class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

num_class = len(class_names)
point_cloud_range = [-7.08, -0.6, -7.5, 7, 9.0, 4.5]  # xyzxyz to voxilize
voxel_size = [0.01, 0.006, 0.3]  # For Loss and Gt calculation

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

model = dict(
    type='MergeNet',
    merge_method='cat',
    merge_in_channels=256,
    img_model_weight='work_dirs/nanodet/model_last.pth',
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(
            16000,  # if training, max_voxels[0],
            20000)),  #  else max_voxels[1]
    voxel_encoder=dict(type='HardSimpleVFE'),
    backbone=dict(
        type='SECONDFPNDCN',
        in_channels=128,
        layer_nums=[3],
        layer_strides=[1],
        num_filters=[128],
        upsample_strides=[2],
        out_channels=[128]),
    pts_backbone=dict(
        type='PointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),  # points for SAMPLER.
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    middle_encoder=dict(
        type='SparseEncoderV2',
        in_channels=4,
        sparse_shape=[40, 1600, 1408],
        out_channels=320),
    bbox_head=dict(
        type='Center3DHead',
        num_classes=num_class,
        in_channels=128,
        feat_channels=128,
        bbox_coder=dict(
            type='Center3DBoxCoder',
            num_class=num_class,
            voxel_size=voxel_size,
            pc_range=point_cloud_range,
            num_dir_bins=0,
            downsample_ratio=4.0,
            min_overlap=0.001,
            keypoint_sensitive=False,
        ),
        loss_cls=dict(type='MSELoss', loss_weight=1.0),
        loss_xy=dict(type='GatherBalancedL1Loss', loss_weight=1.0),
        loss_z=dict(type='GatherBalancedL1Loss', loss_weight=1.0),
        loss_dim=dict(type='GatherBalancedL1Loss', loss_weight=2.0),
        loss_dir=dict(type='GatherBalancedL1Loss', loss_weight=0.5),
        # loss_decode=dict(type='Boxes3dDecodeLoss', loss_weight=0.5),
        bias_cls=-7.94,
        loss_corner=dict(type='MSELoss', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote')),
    test_cfg=dict(
        img_rcnn=dict(score_thr=0.1),
        score_thr=0.1,
        pts=dict(
            sample_mod='seed',
            nms_thr=0.25,
            score_thr=0.05,
            per_class_proposal=True)))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='NanoDetResize', size=(512, 512), keep_ratio=True),
    dict(
        type='ColorAugNorm',
        brightness=0.2,
        contrast=[0.6, 1.4],
        saturation=[0.5, 1.2],
        normalize=[[127.0, 127.0, 127.0], [128.0, 128.0, 128.0]]),
    dict(type='AlignMatrix', align_matrix=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    # dict(type='PointSample', num_points=20000),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'points', 'gt_bboxes_3d',
            'gt_labels_3d'
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='NanoDetResize', size=(512, 512), keep_ratio=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(512, 512),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NanoDetResize', size=(512, 512), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
            # dict(type='PointSample', num_points=20000),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img', 'points'])
        ]),
]

eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img', 'points'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(dataset=dict(pipeline=train_pipeline, filter_empty_gt=True)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
evaluation = dict(pipeline=eval_pipeline)
find_unused_parameters = True
gpu_ids = range(0, 2)