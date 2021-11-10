from mmcv.utils.config import Config
from mmdet.datasets.builder import build_dataloader
from mmdet3d.datasets.builder import build_dataset
import torch

cfg = Config.fromfile(
    '/mmdetection3d/configs/_base_/datasets/coco_instance.py')
dataset = build_dataset(cfg.data.train)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
dataloader = build_dataloader(dataset, 4, 0, dist=False)
for i, data in enumerate(dataloader):
    print(data)