import tqdm
import torch
from typing import Tuple
from mmcv import Config
from mmdet.datasets.builder import build_dataloader


from mmdet3d.datasets.builder import build_dataset

cfg = Config.fromfile('./configs/mergenet/merge_net.py')

datasets = build_dataset(cfg.data.train)
dataloader = build_dataloader(datasets, 1, 1)

with open('empth_files.txt', 'w') as f:
    for i in tqdm.tqdm(dataloader):
        if sum(i['gt_labels_3d'].data[0][0].shape) == 0:
            filename = i['img_metas'].data[0][0]['filename']
            f.writelines(filename+"\n")