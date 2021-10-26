from typing import Tuple
from mmcv import Config

cfg = Config.fromfile('./configs/mergenet/merge_net.py')

print(type(cfg.model))

for key, value in cfg.items():
    if isinstance(value, tuple):
        print(key, value)