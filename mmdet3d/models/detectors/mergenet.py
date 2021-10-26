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
class MergeNet(Base3DDetector):

    def __init__(self,
                 pts_backbone=None,
                 img_backbone=None,
                 img_neck=None,
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

        # image branch
        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.img_bbox_head = builder.build_head(img_bbox_head)

        # Merge Branch(Centernet3d's head)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.centernet3d_head = builder.build_head(bbox_head)

        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, imgs):
        "mmdetection3d needs such abstract method."
        pass

    def extrac_img_feat(self, img, img_metas=None):
        x = self.img_backbone(img)
        img_features = self.img_neck(x)
        img_bbox = self.img_bbox_head(x)
        return img_features, img_bbox

    def extract_pts_faet(self, points):
        x = self.pts_backbone(points)
        seed_points = x['fp_xyz'][-1]
        seed_features = x['fp_features'][-1]
        seed_indices = x['fp_indices'][-1]

        return (seed_points, seed_features, seed_indices)

    def forward_train(self,
                      img,
                      points,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        # img feature
        img_features, img_bbox = self.extrac_img_feat(img)

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

    def simple_test(self, points, img_metas, imgs, rescale=False):
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
        bbox_list = self.bbox_head.get_bboxes(pred_dict, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels, img_meta)
            for bboxes, scores, labels, img_meta in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs, rescale=False):
        img_features, img_bbox = self.extrac_img_feat(imgs)

        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            # points feature
            points = torch.stack(points)
            seeds_3d, seed_3d_features, seed_indices = self.extrac_pts_feat(
                points)

            # merge
            print(img_features.shape, seeds_3d.shape, seed_3d_features.shape,
                  seed_indices)

            x = torch.randn(6, 128, 400, 352)
            pred_dict = self.centernet3d_head(x)
            bbox_list = self.bbox_head.get_bboxes(pred_dict, img_metas)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)
        return merged_bboxes

    def forward_dummy(self, points):
        return super().forward_dummy(points)