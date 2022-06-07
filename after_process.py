import numpy as np
import pickle as pkl
from materials.process_priori import cal_interAUPriori
import csv

pkl_path = '/media/database/data4/wf/AU_EMOwPGM/codes/save/2022-06-07/results.pkl'
with open(pkl_path, 'rb') as fo:
    pkl_file = pkl.load(fo)
ori_EMO2AU, ori_prob_AU, ori_EMO_img_num, ori_AU_cpt, EMO, AU = cal_interAUPriori()
new_EMO2AU = pkl_file['new_AU2EMO']
new_AUOcc = pkl_file['AU_occ']
info_size = pkl_file['train_size']

ori_size = sum(ori_EMO_img_num)
prob_EMO = 1.0 / len(EMO)
new_EMO2AU_tmp = new_EMO2AU/1.0*len(EMO)
new_EMO2AU = []
new_probAU = []
for i, au in enumerate(AU):
    prob_AUi = (new_AUOcc[au] + ori_prob_AU[i]*ori_size) / (ori_size + info_size)
    new_probAU.append(prob_AUi)
    new_EMO2AU.append([au]+list(new_EMO2AU_tmp[i, :]*prob_AUi))
new_EMO2AU = np.array(new_EMO2AU).T
with open('/media/database/data4/wf/AU_EMOwPGM/codes/save/2022-06-07/new_EMO2AU.csv', 'w', encoding='UTF8', newline='') as f: # 写入PGM推理的结果
    csv_writer = csv.writer(f)
    csv_writer.writerows(list(new_EMO2AU))


end_flag = True