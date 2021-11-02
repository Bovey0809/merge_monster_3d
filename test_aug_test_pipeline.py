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
from mmdet3d.core.evaluation.seg_eval import fast_hist

from mmdet3d.datasets.builder import build_dataset
from mmdet3d.models.builder import build_model
from mmcv.parallel import MMDataParallel
import pickle

cfg = Config.fromfile('./configs/mergenet/merge_net.py')

datasets = build_dataset(cfg.data.test)

# For testing
dataloader = build_dataloader(datasets, 6, 1, dist=False, shuffle=False)
model = build_model(cfg.model)
set_random_seed(0)
model = MMDataParallel(model, device_ids=[0])
# show bin file addresss

dataset = dataloader.dataset

with open('./cuda_error_batch', 'rb') as f:
    data = pickle.load(f)

# Test Evalation and Test pipeline
model.eval()
for i, data in tqdm.tqdm(enumerate(dataloader)):
    img = data['img'][0]
    img_metas = data['img_metas'][0]
    points = data['points'][0]
    inputs = dict(img=img, img_metas=img_metas, points=points)
    result = model(return_loss=False, **inputs)
