# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset

from .pipelines import NanodetPipeline


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    A base class of detection dataset. Referring from MMDetection.
    A dataset should have images, annotations and preprocessing pipelines
    NanoDet use [xmin, ymin, xmax, ymax] format for box and
     [[x0,y0], [x1,y1] ... [xn,yn]] format for key points.
    instance masks should decode into binary masks for each instance like
    {
        'bbox': [xmin,ymin,xmax,ymax],
        'mask': mask
     }
    segmentation mask should decode into binary masks for each class.

    :param img_path: image data folder
    :param ann_path: annotation file path or folder
    :param use_instance_mask: load instance segmentation data
    :param use_seg_mask: load semantic segmentation data
    :param use_keypoint: load pose keypoint data
    :param load_mosaic: using mosaic data augmentation from yolov4
    :param mode: train or val or test
    """

    def __init__(
        self,
        img_path,
        ann_path,
        sem_img_path,
        input_size,
        pipeline,
        keep_ratio=True,
        use_instance_mask=False,
        use_seg_mask=False,
        use_keypoint=False,
        load_mosaic=False,
        mode="train",
    ):
        assert mode in ["train", "val", "test"]
        self.img_path = img_path
        self.ann_path = ann_path
        self.sem_img_path = sem_img_path
        self.input_size = input_size
        self.pipeline = NanodetPipeline(pipeline, keep_ratio)
        self.keep_ratio = keep_ratio
        self.use_instance_mask = use_instance_mask
        self.use_seg_mask = use_seg_mask
        self.use_keypoint = use_keypoint
        self.load_mosaic = load_mosaic
        self.mode = mode

        self.data_info = self.get_data_info(ann_path)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if self.mode == "val" or self.mode == "test":
            return self.get_val_data(idx)
        else:
            while True:
                data = self.get_train_data(idx)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    @abstractmethod
    def get_data_info(self, ann_path):
        pass

    @abstractmethod
    def get_train_data(self, idx):
        pass

    @abstractmethod
    def get_val_data(self, idx):
        pass

    def get_another_id(self):
        return np.random.random_integers(0, len(self.data_info) - 1)
