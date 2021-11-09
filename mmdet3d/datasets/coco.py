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
from mmdet.datasets import DATASETS

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
    'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite',
    'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard',
    'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant',
    'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear',
    'hair_drier', 'toothbrush'
]


@DATASETS.register_module()
class COCONanoDetDataset(BaseDataset):

    def __init__(self,
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
                 test_mode=False,
                 mode="train"):
        if test_mode:
            mode = "test"
        else:
            mode = 'train'
        super().__init__(
            img_path,
            ann_path,
            sem_img_path,
            input_size,
            pipeline,
            keep_ratio=keep_ratio,
            use_instance_mask=use_instance_mask,
            use_seg_mask=use_seg_mask,
            use_keypoint=use_keypoint,
            load_mosaic=load_mosaic,
            mode=mode)
        self.CLASSES = CLASSES
        if mode == 'train':
            self._set_group_flag()

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_info[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def get_data_info(self, ann_path):
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id = img_info["id"]
        if not isinstance(id, int):
            raise TypeError("Image id must be int.")
        info = {
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": id
        }
        return info

    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []
        for ann in anns:
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann["keypoints"])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
        if self.use_instance_mask:
            annotation["masks"] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation["keypoints"] = np.array(
                    gt_keypoints, dtype=np.float32)
            else:
                annotation["keypoints"] = np.zeros((0, 51), dtype=np.float32)
        return annotation

    def get_train_data(self, idx):
        #### load original nanodet for box
        ann = self.get_img_annotation(idx)
        img_info = self.get_per_img_info(idx)
        file_name = img_info["file_name"]
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print("image {} read failed.".format(image_path))
            raise FileNotFoundError(
                "Cant load image! Please check image path!")
        meta = dict(
            img=img,
            img_info=img_info,
            gt_bboxes=ann["bboxes"],
            gt_labels=ann["labels"])

        ### load semantic stuff
        image_path = os.path.join(self.sem_img_path, file_name[:-3] + 'png')
        img = np.array(Image.open(image_path))
        if img is None:
            print("semantic image {} read failed.".format(image_path))
            raise FileNotFoundError(
                "Cant load semantic image! Please check semantic image path!")

        img[img == 0] = 255
        img = img + 79
        img[img == 78] = 255

        ### load semantic thing individual
        n = len(ann['masks'])
        for i in range(n):
            img[ann['masks'][i] == 1] = ann["labels"][i]

        meta["img_semantic_stuff"] = img

        ### process images
        meta = self.pipeline(meta, self.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(
            2, 0, 1))  #h,w,c to c,h,w
        meta["img_semantic_stuff"] = torch.from_numpy(
            meta["img_semantic_stuff"]).unsqueeze(0).to(
                dtype=torch.float32)  #h,w

        # ##### check image
        # unique_items = np.unique(meta["img_semantic_stuff"])
        # print('Stuff unique', unique_items)

        # fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(18, 10))
        # for line in axs:
        #     for a in line:
        #         a.axis('off')
        # i = 0
        # ax = axs[i // 2, i % 2]
        # pic1 = ax.imshow(meta["img_semantic_stuff"].numpy())
        # plt.colorbar(pic1)

        # i = 1
        # ax = axs[i // 2, i % 2]
        # ax.imshow(meta["img"].numpy()[0,:,:])

        # # i = 2
        # # ax = axs[i // 2, i % 2]
        # # pic2 = ax.imshow(meta["img_semantic_thing"].numpy())
        # # plt.colorbar(pic2)

        # fig.tight_layout()
        # fig.savefig('check_stuff.png')
        # ######################

        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)


# from .CoCo2017_CatLabelNameColor_StuffThing_Separated import info_separated_dict
# from .file_io import PathManager
# import json

# def load_coco_panoptic_json(json_file, image_dir, gt_dir, meta):
#     """
#     Args:
#         image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
#         gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
#         json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

#     Returns:
#         list[dict]: a list of dicts in Detectron2 standard format. (See
#         `Using Custom Datasets </tutorials/datasets.html>`_ )
#     """

#     def _convert_category_id(segment_info, meta):
#         if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
#             segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
#                 segment_info["category_id"]
#             ]
#             segment_info["isthing"] = True
#         else:
#             segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
#                 segment_info["category_id"]
#             ]
#             segment_info["isthing"] = False
#         return segment_info

#     with PathManager.open(json_file) as f:
#         json_info = json.load(f)

#     ret = []
#     for ann in json_info["annotations"]:
#         image_id = int(ann["image_id"])
#         # TODO: currently we assume image and label has the same filename but
#         # different extension, and images have extension ".jpg" for COCO. Need
#         # to make image extension a user-provided argument if we extend this
#         # function to support other COCO-like datasets.
#         image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
#         label_file = os.path.join(gt_dir, ann["file_name"])
#         segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
#         ret.append(
#             {
#                 "file_name": image_file,
#                 "image_id": image_id,
#                 "pan_seg_file_name": label_file,
#                 "segments_info": segments_info,
#             }
#         )
#     assert len(ret), f"No images found in {image_dir}!"
#     assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
#     assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
#     return ret
# semantic_stuff_info = self.semantic_stuff[idx]
# print(semantic_stuff_info['pan_seg_file_name'] )

# self.semantic_stuff = load_coco_panoptic_json(
#     sem_ann_path, img_path, sem_img_path, info_separated_dict
# ) #from detectron2
