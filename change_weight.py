import torch

state_dict = torch.load('./work_dirs/nanodet/model_last.pth')
params = state_dict['state_dict']

new_paras = dict()
for key, value in params.items():
    if key.startswith('fpn'):
        key = key.replace('fpn', 'neck')
    new_paras[key] = value


torch.save(new_paras, 'weights_changed.pth')
