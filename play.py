# Inorder to test sparse shape of the config
from mmdet3d.ops import spconv as spconv
import torch
import numpy as np

features = np.zeros((2, 4))
features = torch.FloatTensor(features)
coors = np.array([0, 1]).T
coors = torch.IntTensor(coors)
x = spconv.SparseConvTensor(features, coors, (2, 4), 2)
print(x.dense())
