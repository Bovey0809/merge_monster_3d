_base_ = ["../_base_/default_runtime.py", "../_base_/models/nanodet.py"]

dataset_type = 'CocoNanoDetDataset'
data_root = '/semanticfinal_trainmask/data/dataset_coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='SemanticStuff'),
    dict(
        type='WarpResize',
        size=(512, 512),
        perspective=0.0,
        scale=[0.5, 1.5],
        stretch=[[1, 1], [1, 1]],
        rotation=0,
        shear=0,
        translate=0.2,
        flip=0.5),
    dict(
        type='ColorAugNorm',
        brightness=0.2,
        contrast=[0.6, 1.4],
        saturation=[0.5, 1.2],
        normalize=[[127.0, 127.0, 127.0], [128.0, 128.0, 128.0]]),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'img_semantic_stuff', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'warp_matrix')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='SemanticStuff'),
    dict(
        type='ColorAugNorm',
        normalize=[[127.0, 127.0, 127.0], [128.0, 128.0, 128.0]])
]
data = dict(
    samples_per_gpu=40,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        seg_prefix=data_root + 'panoptic_stuff_train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'panoptic_stuff_val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'panoptic_stuff_val2017/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm', 'PQ'], interval=1)

lr = 0.0001
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[10, 20, 35, 70, 90, 110, 130, 150, 180, 210])
runner = dict(type='EpochBasedRunner', max_epochs=280)
total_epochs = 280
find_unused_parameters = True
load_from = 'weights_changed.pth'
