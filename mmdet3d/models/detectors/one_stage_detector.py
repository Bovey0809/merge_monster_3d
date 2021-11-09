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

import time

import torch
import torch.nn as nn
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.detectors import single_stage, FCOS, SingleStageDetector


@DETECTORS.register_module()
class OneStageDetector(BaseModule):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 head_semantic_stuff=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.head_semantic_stuff = build_head(head_semantic_stuff)

    def extract_feat(self, img):
        return super().extract_feat(img)

    def forward(self, x):
        x = self.backbone(x)
        feature128 = x[0]
        feature64 = x[1]
        x = self.neck([x[1], x[2], x[3]])  # feature32
        x_box = self.head(x)
        x_semantic_stuff, x_semantic_thing_mask = self.head_semantic_stuff(
            x[0], feature64, feature128)
        return x_box, x_semantic_stuff, x_semantic_thing_mask

    def inference(self, meta, class_names):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            preds_box, preds_semantic_stuff, preds_semantic_thing_mask = self(
                meta["img"])
            torch.cuda.synchronize()
            time2 = time.time()
            # print("forward time: {:.3f}s".format((time2 - time1)), end=" | ")

            # process box result
            preds_box = self.head.post_process(preds_box, meta)
            preds_box = preds_box[0]

            # process semantic result
            preds_semantic_stuff = self.head_semantic_stuff.post_process(
                preds_semantic_stuff, preds_semantic_thing_mask, meta)

            torch.cuda.synchronize()
            # print("decode time: {:.3f}s".format((time.time() - time2)), end=" | ")
        return (preds_box, preds_semantic_stuff)

    def forward_train(self, gt_meta):
        preds_box, preds_semantic_stuff, preds_semantic_thing_mask = self(
            gt_meta["img"])
        loss_box, loss_states_box = self.head.loss(preds_box, gt_meta)
        loss_semantic_stuff, loss_states_semantic_stuff = self.head_semantic_stuff.loss(
            preds_semantic_stuff, preds_semantic_thing_mask,
            gt_meta["img_semantic_stuff"])

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

            # ThingMask_Dice=loss_states_semantic_stuff['Dice_Loss_thing_mask'],
            Thing_Mask=loss_states_semantic_stuff['Focal_Loss_thing_mask'])

        return (preds_box, preds_semantic_stuff,
                preds_semantic_thing_mask), loss, loss_states
