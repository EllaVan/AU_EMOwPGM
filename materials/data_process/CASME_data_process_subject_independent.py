# 不按subject划分数据集，所有subject的数据混合到一起后再划分数据集
# 每个subject的的每个任务的数据都按照7:3划分
import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import os
import pickle as pkl
import random
import numpy as np
import pandas as pd

from materials.process_priori import cal_interAUPriori

CASME_pkl = {}
label_folder = '/media/data1/wf/AU_EMOwPGM/codes/dataset/CASME/CASME2-coding.xlsx'
img_root_path = '/media/data1/Expression/CASME2/Cropped'
list_path_prefix = '../dataset/CASME2'
data_nodes = ['happy', 'sad', 'fear', 'surprise', 'disgust']#, 'repression']

def get_priori():
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    EMO.remove('anger')
    EMO2AU_cpt1 = EMO2AU_cpt[:2, :]
    EMO2AU_cpt2 = EMO2AU_cpt[3:, :]
    EMO2AU_cpt = np.concatenate([EMO2AU_cpt1, EMO2AU_cpt2])
    ori_size = np.sum(np.array(EMO_img_num))
    num_all_img = ori_size
    AU_cnt = prob_AU * ori_size
    AU_ij_cnt = np.zeros_like(AU_cpt)
    for au_ij in range(AU_cpt.shape[0]):
        AU_ij_cnt[:, au_ij] = AU_cpt[:, au_ij] * AU_cnt[au_ij]
    input_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return input_rules

EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
EMO_img_num = [230] * len(data_nodes)
a = [0, 1, 2, 4, 5]
EMO2AU_cpt = EMO2AU_cpt[a, :]
priori = {'EMO2AU_cpt': EMO2AU_cpt,
        'prob_AU': prob_AU,
        'EMO_img_num': EMO_img_num,
        'AU_cpt': AU_cpt,
        'EMO': EMO,
        'AU': AU}
CASME_pkl['priori'] = priori
priori_rules = get_priori()
CASME_pkl['priori_rules'] = priori_rules

df = pd.read_excel(label_folder, header=0)
new_col = list(range(df.shape[1]))
df.columns = new_col
# a = [2, 6]
# df = df.drop(a, axis=1)
sub_col = 0
file_col = 1
apex_col = 4
AU_col = 7
EMO_col = 8
df_sub = df.iloc[:, sub_col]
df_data = df.iloc[:, file_col]
df_apex = df.iloc[:, apex_col]
df_AU = df.iloc[:, AU_col]
df_label = df.iloc[:, EMO_col]
df = pd.concat([df_sub, df_data, df_apex, df_AU, df_label], axis=1)
new_col = list(range(df.shape[1]))
df.columns = new_col

datalist = {}
EMO = data_nodes
CASME_pkl['EMO'] = EMO
CASME_pkl['AU'] = AU
EMO2index_dict = dict(zip(EMO, range(len(data_nodes)))) #通过名称找到index
for k, emo in enumerate(EMO):
    datalist[k] = {}
    datalist[k]['labelsAU'] = []
    datalist[k]['img_path'] = []

img_sum = 0
for i in range(df.shape[0]):
    item = df.iloc[i, :]
    item_label = item[4]
    if item_label != 'others' and item_label != 'repression' and item_label != 'neutral':
    # if item_label != 'others' and item_label != 'neutral':
        AU_array_tmp = np.zeros((1, 99))
        label = EMO2index_dict[item_label]
        AU_list = str(item[3]).split('+')
        for au in AU_list:
            if au[0] == 'R' or au[0] == 'L':
                au = au[1:]
            au = int(au)
            AU_array_tmp[0, au] = 1
        datalist[label]['labelsAU'].append(AU_array_tmp[0, AU])

        subject = str(item[0]).zfill(2)
        filename = item[1]
        apex = str(item[2])
        file_path = os.path.join(img_root_path, 'sub'+subject, filename, 'reg_img'+apex+'.jpg')
        datalist[label]['img_path'].append(file_path)
        img_sum = img_sum + 1

CASME_pkl['train'] = {}
CASME_pkl['train']['img_path'] = []
CASME_pkl['train']['labelsAU'] = []
CASME_pkl['train']['labelsEMO'] = []
CASME_pkl['train']['EMO_weight'] = []
CASME_pkl['test'] = {}
CASME_pkl['test']['img_path'] = []
CASME_pkl['test']['labelsAU'] = []
CASME_pkl['test']['labelsEMO'] = []
for k, emo in enumerate(EMO):
    a = datalist[k]['img_path']
    b = datalist[k]['labelsAU'] 
    c = list(zip(a, b))
    random.shuffle(c)
    train_len = int(0.9 * len(c))
    datalist_train = c[:train_len]
    for i in range(train_len):
        CASME_pkl['train']['img_path'].append(c[i][0])
        CASME_pkl['train']['labelsAU'].append(c[i][1])
    CASME_pkl['train']['labelsEMO'] += [k]*train_len
    CASME_pkl['train']['EMO_weight'].append(1.0/img_sum * len(c))
    test_len = len(c) - train_len
    for i in range(test_len):
        datalist_test = c[train_len:]
        CASME_pkl['test']['img_path'].append(c[i][0])
        CASME_pkl['test']['labelsAU'].append(c[i][1])
    CASME_pkl['test']['labelsEMO'] += [k]*test_len

CASME_pkl['train']['labelsAU'] = np.array(CASME_pkl['train']['labelsAU'])
CASME_pkl['train']['labelsEMO'] = np.array(CASME_pkl['train']['labelsEMO'])
CASME_pkl['test']['labelsAU'] = np.array(CASME_pkl['test']['labelsAU'])
CASME_pkl['test']['labelsEMO'] = np.array(CASME_pkl['test']['labelsEMO'])

class_num = len(AU)
AUoccur_rate = np.zeros((1, CASME_pkl['train']['labelsAU'].shape[1]))
for i in range(CASME_pkl['train']['labelsAU'].shape[1]):
	AUoccur_rate[0, i] = sum(CASME_pkl['train']['labelsAU'][:,i]>0) / float(CASME_pkl['train']['labelsAU'].shape[0])
AU_weight = 1.0 / AUoccur_rate
AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
le = CASME_pkl['train']['labelsAU'].shape[0]
new_aus = np.zeros((le, class_num * class_num))
for j in range(class_num):
	for k in range(class_num):
		new_aus[:,j*class_num+k] = 2 * CASME_pkl['train']['labelsAU'][:,j] + CASME_pkl['train']['labelsAU'][:,k]
CASME_pkl['train']['AU_weight'] = AU_weight
CASME_pkl['train']['AU_relation'] = new_aus

with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/CASME/CASME_subject_independent.pkl', 'wb') as fo:
    pkl.dump(CASME_pkl, fo)

end_flag = True