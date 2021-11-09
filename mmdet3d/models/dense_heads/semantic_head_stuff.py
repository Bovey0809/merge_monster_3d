from mmdet.models.builder import HEADS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# import json
# import copy
# from ..loss.segment_loss import DiceLoss, FocalLoss


def make_one_hot(labels, classes, clsoffset):
    one_hot = torch.FloatTensor(labels.size()[0],
                                classes).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data - clsoffset, 1)
    return target


class DiceLoss(nn.Module):

    def __init__(self, smooth=1., ignore_index=255, clsoffset=0):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.clsoffset = clsoffset

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(
            target.unsqueeze(dim=1),
            classes=output.size()[1],
            clsoffset=self.clsoffset)
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):

    def __init__(self,
                 gamma=2,
                 alpha=None,
                 ignore_index=255,
                 size_average=True,
                 clsoffset=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(
            reduce=False, ignore_index=ignore_index, weight=alpha)
        self.clsoffset = clsoffset

    def forward(self, output, target):
        logpt = self.CE_loss(output, target - self.clsoffset)
        pt = torch.exp(-logpt)
        loss = ((1 - pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class FocalLoss_BCE(nn.Module):

    def __init__(self,
                 gamma=2,
                 alpha=None,
                 ignore_index=255,
                 size_average=True):
        super(FocalLoss_BCE, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.BCEWithLogitsLoss(reduce=False, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class DepthwiseConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_ch,
                bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x):
        x = self.depthwise(x)
        return x


class PointwiseConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(PointwiseConv, self).__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x):
        x = self.pointwise(x)
        return x


@HEADS.register_module()
class SemanticHeadStuff(nn.Module):

    def __init__(self, in_ch32, in_ch64, in_ch128, hidden_ch, class_ts,
                 droprate):
        super(SemanticHeadStuff, self).__init__()
        self.layer_32 = nn.Sequential(
            DepthwiseConv(in_ch32, in_ch32),
            nn.Dropout(p=droprate),
            PointwiseConv(in_ch32, hidden_ch),
        )

        self.upsample32_64 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.layer_64 = nn.Sequential(
            DepthwiseConv(in_ch64, in_ch64),
            nn.Dropout(p=droprate),
            PointwiseConv(in_ch64, hidden_ch),
        )

        self.upsample64_128 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.layer_128 = nn.Sequential(
            DepthwiseConv(in_ch128, in_ch128),
            nn.Dropout(p=droprate),
            PointwiseConv(in_ch128, hidden_ch),
        )

        self.predictor_thing_stuff = nn.Conv2d(
            hidden_ch, class_ts, kernel_size=1, stride=1, padding=0)
        self.predictor_thing_mask = nn.Conv2d(
            hidden_ch, 1, kernel_size=1, stride=1, padding=0)

        self.dice_loss_stuff = DiceLoss(
            smooth=1., ignore_index=255, clsoffset=80)
        self.dice_loss_thing = DiceLoss(smooth=1., ignore_index=255)

        # weight_thing_stuff = torch.ones(class_ts).float()
        # weight_thing_stuff[80:] = 1/100

        self.focal_loss_stuff = FocalLoss(
            gamma=2,
            alpha=None,
            ignore_index=255,
            size_average=True,
            clsoffset=80)
        self.focal_loss_thing = FocalLoss(
            gamma=2, alpha=None, ignore_index=255, size_average=True)

        self.dice_loss_thing_mask = DiceLoss(smooth=1., ignore_index=255)
        self.focal_loss_thing_mask = FocalLoss_BCE(
            gamma=2, alpha=None, ignore_index=255, size_average=True)

    def forward(self, x_32, x_64, x_128):
        x_32 = self.layer_32(x_32)
        x_32 = self.upsample32_64(x_32)  # in64

        x_64 = self.layer_64(x_64)
        x_64 += x_32
        x_64 = self.upsample64_128(x_64)  # in128

        x_128 = self.layer_128(x_128)
        x_128 += x_64

        x_32 = self.predictor_thing_stuff(x_128)
        x_32_thing_mask = self.predictor_thing_mask(x_128)
        return x_32, x_32_thing_mask

    def loss(self, preds, preds_thing_mask, gt):
        b, c, h, w = preds.size()

        gt = F.interpolate(gt, size=[h, w], mode="nearest")  # 128
        gt = gt.to(dtype=torch.int64)  # convert from float32 to int64
        gt = gt.permute(0, 2, 3, 1)  # size = [b, h, w, 1]
        gt = torch.flatten(gt, 0, -1)  # size = [bhw1]
        # print(gt.size())

        gt_mask_stuff = gt > 79  # isstuff=True
        gt_mask_thing = gt <= 79  # isthing=True
        # print(gt_mask_stuff.size())
        # print(gt_mask_thing.size())

        gt_stuff = gt[gt_mask_stuff]
        gt_thing = gt[gt_mask_thing]
        # print('gt_stuff', gt_stuff.size())
        # print('gt_thing', gt_thing.size())

        preds = preds.permute(0, 2, 3, 1)  # size = [b, h, w, c]
        preds = torch.flatten(preds, 0, -2)  # size = [bhw, c]
        # print(preds.size())

        preds_stuff = preds[gt_mask_stuff]
        preds_stuff = preds_stuff[:, 80:]
        # print('preds_stuff', preds_stuff.size())

        preds_thing = preds[gt_mask_thing]
        preds_thing = preds_thing[:, :80]
        # print('preds_thing', preds_thing.size())

        dloss_stuff = self.dice_loss_stuff(preds_stuff, gt_stuff)
        floss_stuff = self.focal_loss_stuff(preds_stuff, gt_stuff)
        loss_stuff = dloss_stuff + floss_stuff

        dloss_thing = self.dice_loss_thing(preds_thing, gt_thing)
        floss_thing = self.focal_loss_thing(preds_thing, gt_thing)
        loss_thing = dloss_thing + floss_thing

        # get loss for thing mask
        gt_mask_thing = gt_mask_thing.bool().int().to(dtype=torch.float32)

        preds_thing_mask = preds_thing_mask.permute(0, 2, 3,
                                                    1)  # size = [b, h, w, c]
        # preds_thing_mask = torch.flatten(preds_thing_mask, 0, -2) # size = [bhw, c]
        preds_thing_mask = torch.flatten(preds_thing_mask, 0,
                                         -1)  # size = [bhw, c]

        # dloss_thing_mask = self.dice_loss_thing_mask(preds_thing_mask, gt_mask_thing)
        floss_thing_mask = self.focal_loss_thing_mask(preds_thing_mask,
                                                      gt_mask_thing)
        # print(floss_thing_mask)
        # loss_thing_mask = dloss_thing_mask + floss_thing_mask

        loss = loss_stuff + loss_thing + floss_thing_mask

        loss_states = dict(
            loss=loss,
            Dice_Loss_stuff=dloss_stuff,
            Focal_Loss_stuff=floss_stuff,
            Dice_Loss_thing=dloss_thing,
            Focal_Loss_thing=floss_thing,
            # Dice_Loss_thing_mask=dloss_thing_mask,
            Focal_Loss_thing_mask=floss_thing_mask,
        )
        # print(loss_states)
        return loss, loss_states

    def post_process(self, preds, preds_thing_mask, meta):
        b, c, h, w = preds.size()

        # get inverse warp matrix
        warp_matrix = meta["warp_matrix"]
        warp_matrix = np.linalg.inv(warp_matrix)
        width, height = meta['img_info']["height"], meta['img_info']["width"]

        preds_thing_mask = torch.sigmoid(preds_thing_mask)
        preds_stuff_mask = 1 - preds_thing_mask

        preds_thing_mask = preds_thing_mask.expand([-1, 80, -1, -1])
        preds_stuff_mask = preds_stuff_mask.expand([-1, 53, -1, -1])

        preds_mask = torch.cat((preds_thing_mask, preds_stuff_mask), 1)
        # print(preds_mask.size(), preds_mask)

        preds = preds * preds_mask

        # 128 to 512
        preds = F.interpolate(
            preds, scale_factor=4, mode="bilinear")  # mode="nearest"
        # print(preds.size())

        # preds[:, 80:, :, :] = F.softmax(preds[:, 80:, :, :], dim=1)
        # preds[:, :80, :, :] = F.softmax(preds[:, :80, :, :], dim=1)

        preds = F.softmax(preds, dim=1)

        preds = preds.squeeze(0)
        preds = preds.argmax(dim=0)

        preds = preds.cpu().numpy()

        print('Semantic Thing+Sutff Labels', np.unique(preds))
        preds = cv2.warpPerspective(
            preds,
            warp_matrix,
            dsize=tuple([height, width]),
            flags=0,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255))

        return preds

        # ###### process stuff
        # preds_stuff = preds[:, 80:, :, :]
        # preds_stuff = F.softmax(preds_stuff, dim=1)
        # preds_stuff = preds_stuff.squeeze(0)
        # # print(preds_stuff.size())

        # preds_stuff = preds_stuff.argmax(dim=0)
        # # print(preds_stuff.size())

        # preds_stuff = preds_stuff + 80
        # preds_stuff = preds_stuff.cpu().numpy()

        # print('Semantic Stuff Labels', np.unique(preds_stuff))
        # preds_stuff = cv2.warpPerspective(preds_stuff, warp_matrix, dsize=tuple([height, width]), flags=0, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # ###### process thing
        # preds_thing = preds[:, :80, :, :]
        # preds_thing = F.softmax(preds_thing, dim=1)
        # preds_thing = preds_thing.squeeze(0)
        # preds_thing = preds_thing.argmax(dim=0)
        # preds_thing = preds_thing.cpu().numpy()

        # print('Semantic Thing Labels', np.unique(preds_thing))
        # preds_thing = cv2.warpPerspective(preds_thing, warp_matrix, dsize=tuple([height, width]), flags=0, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # return preds_stuff, preds_thing

        # exit()

        # print(preds_stuff_mask.size(), preds_stuff_mask)

        # preds_stuff = F.softmax(preds[:, 80:, :, :], dim=1)
        # print(preds_stuff.size(), preds_stuff)

        # preds_thing = F.softmax(preds[:, :80, :, :], dim=1)
        # # print(preds_thing.size(), preds_thing)

        # b, c, h, w = preds_thing.size()
        # for i in range(c):
        #     preds_thing[:, i, :, :] = preds_thing[:, i, :, :]*(preds_thing_mask.squeeze(0))

        #     print(i)
        #     print(preds_thing[:, i, :, :].size())
        #     print((preds_thing_mask.squeeze(0)).size())

        # exit()

        # preds_stuff_mask = 1 - preds_thing_mask
        # print(preds_stuff_mask.size(), preds_stuff_mask)

        # preds_stuff = F.softmax(preds[:, 80:, :, :], dim=1)
        # print(preds_stuff.size(), preds_stuff)

        # aaa = torch.ones(1,1,128,128)
        # print(aaa)
        # print(aaa-1)

        # exit()

        # 128 to 512
        # preds = F.interpolate(preds, scale_factor=4, mode="bilinear") # mode="nearest"
        # print(preds.size())

        # preds[:, 80:, :, :] = F.softmax(preds[:, 80:, :, :], dim=1)
        # preds[:, :80, :, :] = F.softmax(preds[:, :80, :, :], dim=1)

        # # # preds[:, :80, :, :] = F.softmax(preds[:, :80, :, :], dim=1)
        # # for i in range(0, 80):
        # #     preds[:, i, :, :] = preds[:, i, :, :]*(preds_thing_mask.squeeze(0))

        # # # preds[:, 80:, :, :] = F.softmax(preds[:, 80:, :, :], dim=1)
        # # for i in range(80, c):
        # #     preds[:, i, :, :] = preds[:, i, :, :]*(preds_stuff_mask.squeeze(0))

        # # print(preds.size())

        # preds = F.softmax(preds, dim=1)

        # preds = F.interpolate(preds, scale_factor=4, mode="bilinear") # mode="nearest"

        # preds = preds.argmax(dim=1)

        # # print(preds.size())

        # preds = preds.squeeze(0)

        # preds = preds.cpu().numpy()

        # print('Semantic Thing+Sutff Labels', np.unique(preds))
        # preds = cv2.warpPerspective(preds, warp_matrix, dsize=tuple([height, width]), flags=0, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # preds_thing_mask = preds_thing_mask.expand([-1, 133, -1, -1])

        # preds_thing_mask[:, 80:, :, :] = 1 - preds_thing_mask[:, 80:, :, :]