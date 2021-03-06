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
# limitations under the License
import numpy as np
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector
import torch

@DETECTORS.register_module()
class NanoDetMagic(BaseDetector):

    def __init__(self,
                 img_backbone,
                 img_neck=None,
                 img_bbox_head=None,
                 head_semantic_stuff=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(NanoDetMagic, self).__init__()
        self.backbone = build_backbone(img_backbone)
        self.neck = build_neck(img_neck)
        self.head = build_head(img_bbox_head)
        self.head_semantic_stuff = build_head(head_semantic_stuff)

    def extract_feat(self, img):
        x = self.backbone(img)
        feature128 = x[0]
        feature64 = x[1]
        feature32 = self.neck([x[1], x[2], x[3]])  # feature32
        return feature32, feature64, feature128

    def _forward(self, img):
        feature_32, feature64, feature128 = self.extract_feat(img)
        x_box = self.head(feature_32)
        x_semantic_stuff, x_semantic_thing_mask = self.head_semantic_stuff(
            feature_32[0], feature64, feature128)
        return x_box, x_semantic_stuff, x_semantic_thing_mask

    def forward_train(self, img, img_metas, **gt):
        device = img.device
        preds_box, preds_semantic_stuff, preds_semantic_thing_mask = self._forward(
            img)
        loss_box, loss_states_box = self.head.loss(preds_box, gt)
        gt_masks = gt['img_semantic_stuff'].to(device)
        loss_semantic_stuff, loss_states_semantic_stuff = self.head_semantic_stuff.loss(
            preds_semantic_stuff, preds_semantic_thing_mask, gt_masks)

        loss = loss_box + loss_semantic_stuff

        loss_states = dict(
            loss=loss,
            Box_QFL=loss_states_box['loss_qfl'],
            Box_Bbox=loss_states_box['loss_bbox'],
            Box_DFL=loss_states_box['loss_dfl'],
            Stuff_Dice=loss_states_semantic_stuff['Dice_Loss_stuff'],
            Stuff_Focal=loss_states_semantic_stuff['Focal_Loss_stuff'],
            Thing_Dice=loss_states_semantic_stuff['Dice_Loss_thing'],
            Thing_Focal=loss_states_semantic_stuff['Focal_Loss_thing'],
            Thing_Mask=loss_states_semantic_stuff['Focal_Loss_thing_mask'])
        return loss_states

    def forward_test(self, imgs, img_metas, **kwargs):
        ids = [meta['img_info']['id'] for meta in img_metas]
        for idx in ids:
            pass
        feature_32, _, _ = self.extract_feat(imgs)
        preds_box = self.head(feature_32)
        det_results = self.head.post_process(preds_box, img_metas)
        return det_results

    def simple_test(self, img, img_metas, **kwargs):
        return super().simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return super().aug_test(imgs, img_metas, **kwargs)