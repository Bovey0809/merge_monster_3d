from typing import Tuple
from mmcv import Config

cfg = Config.fromfile('./configs/centernet3d_debug.py')

print(type(cfg.model))

for key, value in cfg.items():
    if not isinstance(value, tuple):
        print(key, value)