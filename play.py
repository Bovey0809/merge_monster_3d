import numpy as np
import torch
fmap = np.load('./fmap.npy')
index = np.load('./index.npy')

torch.tensor(fmap).gather(dim=1, index=torch.tensor(index))