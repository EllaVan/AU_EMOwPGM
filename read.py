import torch

path = '/media/data1/wf/AU_EMOwPGM/codes/results/RAF-DB-compound/Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/epoch1_model_fold0.pth'
info = torch.load(path, map_location='cpu')
# output_rules = info['output_rules']
# val_confu_m = info['val_info']['val_confu_m']
# for i in range(val_confu_m.shape[0]):
#     val_confu_m[:, i] = val_confu_m[:, i] / val_confu_m[:, i].sum(axis=0) * 100.00
a = 1