import time
from mmdet.apis.train import set_random_seed
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
import pickle
cfg = Config.fromfile('./configs/mergenet/merge_net.py')

datasets = build_dataset(cfg.data.train)
dataloader = build_dataloader(datasets, 6, 1, dist=False, shuffle=False)
model = build_model(cfg.model)
set_random_seed(0)
model = MMDataParallel(model, device_ids=[0])
# show bin file addresss

dataset = dataloader.dataset

with open('./cuda_error_batch', 'rb') as f:
    data = pickle.load(f)

result = model(return_loss=True, **data)