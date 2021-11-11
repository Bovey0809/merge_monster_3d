_base_ = [
    "../_base_/schedules/cyclic_20e.py", "../_base_/default_runtime.py",
    "../_base_/models/nanodet.py"
]

dataset_type = 'CocoDataset'
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
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'img_semantic_stuff', 'gt_bboxes', 'gt_labels', 'gt_masks'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
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
