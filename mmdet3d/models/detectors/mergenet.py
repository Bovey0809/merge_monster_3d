# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet3d.ops.voxel.voxelize import Voxelization
import torch.nn as nn
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


@DETECTORS.register_module()
class MergeNet(Base3DDetector):

    def __init__(self,
                 voxel_layer=None,
                 voxel_encoder=None,
                 pts_backbone=None,
                 img_backbone=None,
                 img_head=None,
                 img_neck=None,
                 img_bbox_head=None,
                 backbone=None,
                 middle_encoder=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 img_pretrained=None,
                 head_semantic_stuff=None,
                 merge_method=None,
                 merge_in_channels=None,
                 pretrained=None,
                 init_cfg=None,
                 magic_merge=None,
                 normal_style=None,
                 upsampe_style=None,
                 merge_style=None,
                 img_model_weight=None):
        super(MergeNet, self).__init__(init_cfg=init_cfg)

        # self.merge_method = merge_method
        # point branch
        if pts_backbone is not None:
            self.pts_backbone = builder.build_backbone(pts_backbone)

        # image branch
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)
        if img_head:
            self.img_head = builder.build_head(img_head)
        if head_semantic_stuff:
            self.head_semantic_sutff = builder.build_head(head_semantic_stuff)
        if img_bbox_head:
            self.img_bbox_head = builder.build_head(img_bbox_head)
        self.freeze_img_branch_params()

        # Merge Branch(Centernet3d's head)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)

        self.backbone = builder.build_backbone(backbone)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.centernet3d_head = builder.build_head(bbox_head)

        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # if img_pretrained:
        #     state_dict = torch.load(img_pretrained)
        #     self.load_state_dict(state_dict=state_dict, strict=False)
        self.magic_merge = magic_merge
        self.normal_style = normal_style
        self.upsampe_style = upsampe_style
        self.merge_style = merge_style
        self.img_model_weight = img_model_weight
        self.merge_conv2d = nn.Sequential(
            nn.Conv2d(224, 128, 3, 1, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU())
        self.BN2D_point = nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True)
        self.BN2D_image = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.1, affine=True)

    def extract_feat(self, imgs):
        "mmdetection3d needs such abstract method."
        pass

    def freeze_img_branch_params(self):
        """Freeze all image branch parameters."""
        if self.with_img_bbox_head:
            for param in self.img_bbox_head.parameters():
                param.requires_grad = False
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
        if self.with_img_rpn:
            for param in self.img_rpn_head.parameters():
                param.requires_grad = False
        if self.with_img_roi_head:
            for param in self.img_roi_head.parameters():
                param.requires_grad = False

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return ((hasattr(self, 'img_roi_head') and self.img_roi_head.with_bbox)
                or (hasattr(self, 'img_bbox_head')
                    and self.img_bbox_head is not None))

    @property
    def with_img_bbox_head(self):
        """bool: Whether the detector has a 2D image box head (not roi)."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @torch.no_grad()
    def extrac_img_feat(self, img, img_metas=None):
        x = self.img_backbone(img)
        if self.img_neck._get_name() == 'TAN':
            img_features = self.img_neck([x[1], x[2], x[3]])  # feature32
        else:
            img_features = self.img_neck(x)
        img_bbox = self.img_bbox_head(img_features)
        return img_features, img_bbox

    def extract_pts_feat(self, points):
        x = self.pts_backbone(points)
        seed_points = x['fp_xyz'][-1]
        seed_features = x['fp_features'][-1]
        seed_indices = x['fp_indices'][-1]

        return (seed_points, seed_features, seed_indices)

    @torch.no_grad()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_voxel_feat(self, points):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        point_misc = None
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        # print("x shape",x[0].shape)
        # if xconv2 is not None:
        #     x=[x[0]+xconv2]
        return x, point_misc

    # def merge(self, img_features, point_feat, method='cat'):
    #     assert method in ('cat', 'upsample', 'deconv')
    #     # TODO Add different method for trail.
    #     shape = point_feat[0].shape
    #     new_img_features = F.interpolate(
    #         img_features[0], size=[shape[2], shape[3]])
    #     merged_feature0 = torch.cat((new_img_features, point_feat[0]), 1)
    #     merged_feature = self.magic_merge(merged_feature0)
    #     return merged_feature

    def merge_features(self, img_features, point_feat):
        """merge image_features and point_cloud_features with different methods, such as concat, sum, multiply,
        before merging features, all features are normalized to be [-1,1]"""
        # self.BN2D_point=torch.nn.BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True)
        # self.BN2D_image=torch.nn.BatchNorm2d(96,eps=1e-05,momentum=0.1,affine=True)

        shape = point_feat.shape

        if self.normal_style == 'BN':
            img_features = self.BN2D_image(img_features)
            point_feat = self.BN2D_point(point_feat)

        if self.upsampe_style == 'interpolate':
            new_img_features = F.interpolate(
                img_features, size=[shape[2], shape[3]])

        if self.merge_style == 'concat':
            merged_feature0 = torch.cat((new_img_features, point_feat), 1)
            merged_feature = self.merge_conv2d(merged_feature0)
        elif self.merge_style == 'sum':
            merged_feature0 = torch.add(
                new_img_features, point_feat, alpha=self.weight_params)
            merged_feature = self.merge_conv2d(merged_feature0)
        elif self.merge_style == 'multiply':
            a = 0

        return [merged_feature]

    def forward_train(self,
                      img,
                      points=None,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_bboxes_ignore=None):
        # img feature
        img_features, img_bbox = self.extrac_img_feat(img)

        # points feature
        # points = torch.stack(points)

        # seeds_3d, seed_3d_features, seed_indices = self.extract_pts_feat(
        #     points)

        # x, _ = self.extract_voxel_feat(seeds_3d)

        # For points only.
        # points = torch.stack(points)
        # TODO Debug dim change
        # points = [torch.tensor(np.load('./bug_points.npy')).to('cuda')]
        point_features, _ = self.extract_voxel_feat(points)

        # merge
        if self.magic_merge:
            x = self.merge_features(img_features[-1], point_features[0])
        else:
            x = point_features

        pred_dict = self.centernet3d_head(x)

        losses = dict()
        head_loss = self.centernet3d_head.loss(pred_dict, gt_labels_3d,
                                               gt_bboxes_3d)
        losses.update(head_loss)
        return losses

    def simple_test(self, points, img_metas, imgs, rescale=False):
        """Testing for one img and one point cloud.
        """
        # img feature
        img_features, img_bbox = self.extrac_img_feat(imgs)

        # points feature
        # points = torch.stack(points)
        # x, _ = self.extract_voxel_feat(points)
        # merge
        # pred_dict = self.centernet3d_head(x)
        # seeds_3d, seed_3d_features, seed_indices = self.extrac_pts_feat(points)

        # merge
        point_features, _ = self.extract_voxel_feat(points=points)

        # merge
        if self.magic_merge:
            x = self.merge_features(img_features[-1], point_features[0])
        else:
            x = point_features

        pred_dict = self.centernet3d_head(x)
        bbox_list = self.centernet3d_head.get_bboxes(pred_dict, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels, img_meta)
            for bboxes, scores, labels, img_meta in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs, rescale=False):
        feats, _ = self.extract_voxel_feat(points)
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            # points feature
            outs = self.centernet3d_head([x])
            bbox_list = self.centernet3d_head.get_bboxes(outs, img_meta)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels, img_meta in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, [[img_meta]],
                                            self.centernet3d_head.test_cfg)
        return merged_bboxes

    def forward_dummy(self, points, img_metas, imgs):
        pass

    def init_weights(self):
        if self.img_model_weight:
            state_dict = torch.load(self.img_model_weight)['state_dict']
            self.load_state_dict(state_dict=state_dict, strict=False)