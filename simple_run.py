import time
import tqdm
from mmcv.cnn.builder import build_model_from_cfg
import tqdm
import torch
from typing import Tuple
from mmcv import Config
from mmdet.datasets.builder import build_dataloader
from mmdet3d.apis.test import single_gpu_test

from mmdet3d.datasets.builder import build_dataset
from mmdet3d.models.builder import build_model
from mmcv.parallel import MMDataParallel

cfg = Config.fromfile('./configs/mergenet/merge_net.py')

datasets = build_dataset(cfg.data.test)
dataloader = build_dataloader(datasets, 1, 1, dist=False, shuffle=False)
model = build_model(cfg.model)

model = MMDataParallel(model, device_ids=[0])
# show bin file addresss
model.eval()
dataset = dataloader.dataset
with torch.no_grad():
    for i, data in enumerate(dataloader):
        result = model(return_loss=False, rescale=True, **data)