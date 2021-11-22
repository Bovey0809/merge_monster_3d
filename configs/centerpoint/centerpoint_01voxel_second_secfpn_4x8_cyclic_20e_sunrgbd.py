_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_nus.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

voxel_size = [0.01, 0.0066, 0.3]
point_cloud_range = [-7.2, -0.504, -7.5, 7.2, 9.0, 4.8]  # xyzxyz to voxilize
sparse_shape = [40, 1440, 1440]

model = dict(
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(30000, 50000)),
    pts_voxel_encoder=dict(num_features=4),
    pts_middle_encoder=dict(in_channels=4, sparse_shape=[41, 1440, 1440]),
    pts_bbox_head=dict(
        tasks=[
            dict(num_class=1, class_names=['bed']),
            dict(num_class=2, class_names=['table', 'sofa']),
            dict(num_class=2, class_names=['chair', 'toilet']),
            dict(num_class=1, class_names=['desk']),
            dict(num_class=2, class_names=['dresser', 'night_stand']),
            dict(num_class=2, class_names=['bookshelf', 'bathtub']),
        ],
        bbox_coder=dict(pc_range=point_cloud_range,
                        voxel_size=voxel_size[:2])),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1600, 40],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=point_cloud_range,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2])))

dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'sunrgbd_infos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            bed=5,
            table=5,
            sofa=5,
            chair=5,
            toilet=5,
            desk=5,
            dresser=5,
            night_stand=5,
            bookshelf=5,
            bathtub=5)),
    classes=class_names,
    sample_groups=dict(
        bed=2,
        table=3,
        sofa=7,
        chair=4,
        toilet=6,
        desk=2,
        dresser=6,
        night_stand=6,
        bookshelf=2,
        bathtub=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D'),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
