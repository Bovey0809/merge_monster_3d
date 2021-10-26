# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import warnings
from .single_stage import SingleStage3DDetector
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.utils import MLP
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


@DETECTORS.register_module()
class MergeNet(SingleStage3DDetector):

    def __init__(self,
                 pts_backbone=None,
                 pts_bbox_heads=None,
                 pts_neck=None,
                 img_backbone=None,
                 img_neck=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 img_bbox_head=None,
                 middle_encoder=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MergeNet, self).__init__(init_cfg=init_cfg)

        # point branch
        if pts_backbone is not None:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_heads is not None:
            pts_bbox_head_common = pts_bbox_heads.common
            pts_bbox_head_common.update(
                train_cfg=train_cfg.pts if train_cfg is not None else None)
            pts_bbox_head_common.update(test_cfg=test_cfg.pts)
            pts_bbox_head_joint = pts_bbox_head_common.copy()
            pts_bbox_head_joint.update(pts_bbox_heads.joint)
            pts_bbox_head_pts = pts_bbox_head_common.copy()
            pts_bbox_head_pts.update(pts_bbox_heads.pts)
            pts_bbox_head_img = pts_bbox_head_common.copy()
            pts_bbox_head_img.update(pts_bbox_heads.img)

            self.pts_bbox_head_joint = builder.build_head(pts_bbox_head_joint)
            self.pts_bbox_head_pts = builder.build_head(pts_bbox_head_pts)
            self.pts_bbox_head_img = builder.build_head(pts_bbox_head_img)
            self.pts_bbox_heads = [
                self.pts_bbox_head_joint, self.pts_bbox_head_pts,
                self.pts_bbox_head_img
            ]
            self.loss_weights = pts_bbox_heads.loss_weights

        # image branch
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            rpn_train_cfg = train_cfg.img_rpn if train_cfg \
                is not None else None
            img_rpn_head_ = img_rpn_head.copy()
            img_rpn_head_.update(
                train_cfg=rpn_train_cfg, test_cfg=test_cfg.img_rpn)
            self.img_rpn_head = builder.build_head(img_rpn_head_)
        if img_roi_head is not None:
            rcnn_train_cfg = train_cfg.img_rcnn if train_cfg \
                is not None else None
            img_roi_head.update(
                train_cfg=rcnn_train_cfg, test_cfg=test_cfg.img_rcnn)
            self.img_roi_head = builder.build_head(img_roi_head)
        if img_bbox_head is not None:
            self.img_bbox_head = builder.build_head(img_bbox_head)

        # Merge Branch(Centernet3d's head)
        self.centernet3d_head = builder.build_head(bbox_head)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

    def extrac_img_feat(self, img, img_metas):
        x = self.img_backbone(img)
        img_features = self.img_neck(x)
        img_bbox = self.img_bbox_head(x)
        return img_features, img_bbox

    def extract_pts_faet(self, points):
        x = self.pts_backbone(pts)
        x = self.pts_neck(x)
        seed_points = x['fp_xyz'][-1]
        seed_features = x['fp_features'][-1]
        seed_indices = x['fp_indices'][-1]

        return (seed_points, seed_features, seed_indices)

    def forward_train(self,
                      imgs,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        # img feature
        img_features, img_bbox = self.extrac_img_feat(imgs)

        # points feature
        points = torch.stack(points)
        seeds_3d, seed_3d_features, seed_indices = self.extrac_pts_feat(points)

        # merge
        print(img_features.shape, seeds_3d.shape, seed_3d_features.shape,
              seed_indices)

        x = torch.randn(6, 128, 400, 352)
        pred_dict = self.centernet3d_head(x)
        losses = dict()
        head_loss = self.centernet3d_head.loss(pred_dict, gt_labels_3d,
                                               gt_bboxes_3d)
        losses.update(head_loss)
        return losses
