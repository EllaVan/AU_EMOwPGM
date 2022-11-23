import torch

path = '/media/data1/wf/AU_EMOwPGM/codes/extending/save/seen/2022-10-31/AffectNet/output.pth'
info = torch.load(path, map_location='cpu')
output_rules = info['output_rules']
val_confu_m = info['val_info']['val_confu_m']
for i in range(val_confu_m.shape[0]):
    val_confu_m[:, i] = val_confu_m[:, i] / val_confu_m[:, i].sum(axis=0) * 100.00
a = 1