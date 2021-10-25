lr = 0.000225
optimizer = dict(
    type='AdamW', lr=0.000225, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=1)
checkpoint_config = dict(interval=2)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/centernet3d'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'KittiDataset'
data_root = '/mmdetection3d/data/kitti/'
class_names = ['Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
voxel_size = [0.05, 0.05, 0.1]
num_class = 1
evaluation = dict(interval=1)
total_epochs = 1
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root='/mmdetection3d/data/kitti/',
    info_path='/mmdetection3d/data/kitti/kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=['Car'],
    sample_groups=dict(Car=15))
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=dict(backend='disk')),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root='/mmdetection3d/data/kitti/',
            info_path='/mmdetection3d/data/kitti/kitti_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
            classes=['Car'],
            sample_groups=dict(Car=15))),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    dict(
        type='ObjectRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=['Car']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Car'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='KittiDataset',
            data_root='/mmdetection3d/data/kitti/',
            ann_file='/mmdetection3d/data/kitti/kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='ObjectSample',
                    db_sampler=dict(
                        data_root='/mmdetection3d/data/kitti/',
                        info_path=
                        '/mmdetection3d/data/kitti/kitti_dbinfos_train.pkl',
                        rate=1.0,
                        prepare=dict(
                            filter_by_difficulty=[-1],
                            filter_by_min_points=dict(Car=5)),
                        classes=['Car'],
                        sample_groups=dict(Car=15))),
                dict(
                    type='ObjectNoise',
                    num_try=100,
                    translation_std=[1.0, 1.0, 0.5],
                    global_rot_range=[0.0, 0.0],
                    rot_range=[-0.78539816, 0.78539816]),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05]),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(type='PointShuffle'),
                dict(type='DefaultFormatBundle3D', class_names=['Car']),
                dict(
                    type='Collect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            modality=dict(use_lidar=True, use_camera=False),
            classes=['Car'],
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type='KittiDataset',
        data_root='/mmdetection3d/data/kitti/',
        ann_file='/mmdetection3d/data/kitti/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Car'],
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='KittiDataset',
        data_root='/mmdetection3d/data/kitti/',
        ann_file='/mmdetection3d/data/kitti/kitti_infos_test.pkl',
        split='testing',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Car'],
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='CenterNet3D',
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        voxel_size=[0.05, 0.05, 0.1],
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoderV2',
        in_channels=4,
        sparse_shape=[40, 1600, 1408],
        out_channels=320),
    backbone=dict(
        type='SECONDFPNDCN',
        in_channels=128,
        layer_nums=[3],
        layer_strides=[1],
        num_filters=[128],
        upsample_strides=[2],
        out_channels=[128]),
    bbox_head=dict(
        type='Center3DHead',
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        bbox_coder=dict(
            type='Center3DBoxCoder',
            num_class=1,
            voxel_size=[0.05, 0.05, 0.1],
            pc_range=[0, -40, -3, 70.4, 40, 1],
            num_dir_bins=0,
            downsample_ratio=4.0,
            min_overlap=0.01,
            keypoint_sensitive=True),
        loss_cls=dict(type='MSELoss', loss_weight=1.0),
        loss_xy=dict(type='GatherBalancedL1Loss', loss_weight=1.0),
        loss_z=dict(type='GatherBalancedL1Loss', loss_weight=1.0),
        loss_dim=dict(type='GatherBalancedL1Loss', loss_weight=2.0),
        loss_dir=dict(type='GatherBalancedL1Loss', loss_weight=0.5),
        bias_cls=-7.94,
        loss_corner=dict(type='MSELoss', loss_weight=1.0)))
train_cfg = dict()
test_cfg = dict(score_thr=0.1)
find_unused_parameters = True
gpu_ids = range(0, 1)
