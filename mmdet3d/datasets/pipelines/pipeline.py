
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
import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES

from .color import color_aug_and_norm
from .warp import get_resize_matrix, get_translate_matrix, warp_and_resize


@PIPELINES.register_module()
class ColorAugNorm(object):
    """Color and Norm
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(self, input_dict):
        meta = color_aug_and_norm(input_dict, kwargs=self.kwargs)
        return meta

    def __repr__(self) -> str:
        return super().__repr__()


@PIPELINES.register_module()
class WarpResize(object):
    """Warp and Resize follow Nanodet.

    Note: Details are different, gt_masks will be changed during this warp.
    """

    def __init__(self, size, **warp_kwargs) -> None:
        self.dst_shape = size
        self.warp_kwargs = warp_kwargs

    def __call__(self, input_dict: dict) -> dict:
        input_dict = warp_and_resize(
            input_dict, self.warp_kwargs, self.dst_shape, keep_ratio=True)
        return input_dict

    def __repr__(self) -> str:
        return super().__repr__()


@PIPELINES.register_module()
class SemanticStuff(object):
    """Semantic Stuff followed by LianFeng's magic nanodet.
    
    SemanticStuff is one of the groundtruth labels.(For loss calculation only).
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


@PIPELINES.register_module()
class NanoDetResize:

    def __init__(self, size, keep_ratio=True) -> None:
        self.dst_shape = size
        self.keep_ratio = keep_ratio

    def __call__(self, input_dict):
        raw_img = input_dict['img']
        height = raw_img.shape[0]
        width = raw_img.shape[1]

        # Center
        C = np.eye(3)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2
        T = get_translate_matrix(0, width, height)
        M = T @ C
        ResizeM = get_resize_matrix((width, height), self.dst_shape,
                                    self.keep_ratio)
        M = ResizeM @ M
        img = cv2.warpPerspective(raw_img, M, dsize=tuple(self.dst_shape))
        input_dict['img'] = img
        input_dict['warp_matrix'] = M
        return input_dict