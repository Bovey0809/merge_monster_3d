# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from collections import OrderedDict
from os import path as osp

from mmdet3d.core import show_multi_modality_result, show_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.core import eval_map
from mmdet.datasets import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class MyDataset(Custom3DDataset):
    CLASSES = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=True, use_lidar=True),
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False):
        super().__init__(
            data_root,
            ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
