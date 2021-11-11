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

import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

from .base import BaseDataset
from PIL import Image
from mmdet.datasets import DATASETS, CustomDataset, CocoPanopticDataset, CocoDataset
from .custom_3d import Custom3DDataset


@DATASETS.register_module()
class COCONanoDetDataset(CocoDataset):
    # Get the img & add "img_semantic_stuff" field.
    pass