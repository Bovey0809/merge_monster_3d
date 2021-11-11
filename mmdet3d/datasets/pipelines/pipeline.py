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

import functools

from mmdet.datasets.builder import PIPELINES

from .color import color_aug_and_norm
from .warp import warp_and_resize
import cv2
import numpy as np


@PIPELINES.register_module()
class ColorAugNorm(object):
    pass


@PIPELINES.register_module()
class WarpResize(object):

    def __init__(self, size, **warp_kwargs) -> None:
        self.dst_shape = size
        self.warp_kwargs = warp_kwargs

    def __call__(self, input_dict: dict) -> dict:
        input_dict = warp_and_resize(
            input_dict, self.warp_kwargs, self.size, keep_ratio=True)
        return input_dict

    def __repr__(self) -> str:
        return super().__repr__()


@PIPELINES.register_module()
class SemanticStuff(object):
    """Semantic Stuff followed by LianFeng's magic nanodet.
    """

    def __init__(self) -> None:
        super().__init__()

    def _create_seg(self, img, ann):
        """Create segmantic for img.

        Args:
            img (np.array): hxw read from file.
        """

        img[img == 0] = 255
        img = img + 79
        img[img == 78] = 255

        n = len(ann['masks'])
        # TODO: Need optimization.
        for i in range(n):
            img[ann['masks'][i] == 1] = ann['labels'][i]
        return img

    def __call__(self, input_dict):
        seg_img = input_dict['gt_semantic_seg']
        ann = input_dict['ann_info']
        seg_img = self._create_seg(seg_img, ann)
        input_dict['img_semantic_stuff'] = seg_img
        return input_dict

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'Generate Semantic filed.'
        return repr_str


@PIPELINES.register_module()
class NanodetPipeline:

    def __init__(self, cfg, keep_ratio):
        self.warp = functools.partial(
            warp_and_resize, warp_kwargs=cfg, keep_ratio=keep_ratio)
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, meta, dst_shape):
        meta = self.warp(meta=meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta
