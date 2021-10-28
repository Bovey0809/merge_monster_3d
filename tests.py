import tqdm
import torch
from typing import Tuple
from mmcv import Config
from mmdet.datasets.builder import build_dataloader


from mmdet3d.datasets.builder import build_dataset

cfg = Config.fromfile('./configs/mergenet/merge_net.py')

datasets = build_dataset(cfg.data.train)
dataloader = build_dataloader(datasets, 1, 1)

# show bin file addresss
for i in tqdm.tqdm(dataloader):
    

# Compare SUNRGBD to kitti.
