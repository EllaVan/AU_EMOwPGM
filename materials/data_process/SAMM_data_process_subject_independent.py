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

num_list = ['0','1','2','3','4','5','6','7','8','9']
SAMM_pkl = {}
label_folder = '/media/data1/wf/AU_EMOwPGM/codes/dataset/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'
img_root_path = '/media/data1/Expression/SAMM'
list_path_prefix = '../dataset/SAMM'
# data_nodes = ['happy', 'sad', 'fear', 'surprise', 'disgust']#, 'repression']
# data_nodes = ['happy', 'sad', 'anger', 'surprise', 'fear', 'disgust']
data_nodes = ['Happiness', 'Sadness', 'Anger', 'Surprise', 'Fear', 'Disgust']

def get_priori():
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
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
SAMM_pkl['priori'] = priori
priori_rules = get_priori()
SAMM_pkl['priori_rules'] = priori_rules

df = pd.read_excel(label_folder, header=0)
new_col = list(range(df.shape[1]))
df.columns = new_col
# a = [2, 6]
# df = df.drop(a, axis=1)
sub_col = 0
file_col = 1
apex_col = 4
AU_col = 8
EMO_col = 9
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
SAMM_pkl['EMO'] = EMO
SAMM_pkl['AU'] = AU
EMO2index_dict = dict(zip(EMO, range(len(data_nodes)))) #通过名称找到index
for k, emo in enumerate(EMO):
    datalist[k] = {}
    datalist[k]['labelsAU'] = []
    datalist[k]['img_path'] = []

img_sum = 0
for i in range(df.shape[0]):
    item = df.iloc[i, :]
    item_label = item[4]
    # if item_label != 'Other' or item_label != 'Contempt':
    if item_label in data_nodes:
    # if item_label != 'others' and item_label != 'neutral':
        AU_array_tmp = np.zeros((1, 99))
        label = EMO2index_dict[item_label]
        AU_list = str(item[3]).split('+')
        for au in AU_list:
            au_list = list(au)
            mr = []
            for n in au_list:
                if n in num_list:
                    mr.append(n)
            au = int(''.join(list(mr)))
            # if au[0] == 'R' or au[0] == 'L':
            #     au = au[1:]
            # au = int(au)
            AU_array_tmp[0, au] = 1
        datalist[label]['labelsAU'].append(AU_array_tmp[0, AU])

        subject = str(item[0]).zfill(2)
        filename = item[1]
        apex = str(item[2])
        file_path = os.path.join(img_root_path, 'sub'+subject, filename, 'reg_img'+apex+'.jpg')
        datalist[label]['img_path'].append(file_path)
        img_sum = img_sum + 1

SAMM_pkl['train'] = {}
SAMM_pkl['train']['img_path'] = []
SAMM_pkl['train']['labelsAU'] = []
SAMM_pkl['train']['labelsEMO'] = []
SAMM_pkl['train']['EMO_weight'] = []
SAMM_pkl['test'] = {}
SAMM_pkl['test']['img_path'] = []
SAMM_pkl['test']['labelsAU'] = []
SAMM_pkl['test']['labelsEMO'] = []
for k, emo in enumerate(EMO):
    a = datalist[k]['img_path']
    b = datalist[k]['labelsAU'] 
    c = list(zip(a, b))
    random.shuffle(c)
    train_len = int(0.9 * len(c))
    datalist_train = c[:train_len]
    for i in range(train_len):
        SAMM_pkl['train']['img_path'].append(c[i][0])
        SAMM_pkl['train']['labelsAU'].append(c[i][1])
    SAMM_pkl['train']['labelsEMO'] += [k]*train_len
    SAMM_pkl['train']['EMO_weight'].append(1.0/img_sum * len(c))
    test_len = len(c) - train_len
    for i in range(test_len):
        datalist_test = c[train_len:]
        SAMM_pkl['test']['img_path'].append(c[i][0])
        SAMM_pkl['test']['labelsAU'].append(c[i][1])
    SAMM_pkl['test']['labelsEMO'] += [k]*test_len

SAMM_pkl['train']['labelsAU'] = np.array(SAMM_pkl['train']['labelsAU'])
SAMM_pkl['train']['labelsEMO'] = np.array(SAMM_pkl['train']['labelsEMO'])
SAMM_pkl['test']['labelsAU'] = np.array(SAMM_pkl['test']['labelsAU'])
SAMM_pkl['test']['labelsEMO'] = np.array(SAMM_pkl['test']['labelsEMO'])

class_num = len(AU)
AUoccur_rate = np.zeros((1, SAMM_pkl['train']['labelsAU'].shape[1]))
for i in range(SAMM_pkl['train']['labelsAU'].shape[1]):
	AUoccur_rate[0, i] = sum(SAMM_pkl['train']['labelsAU'][:,i]>0) / float(SAMM_pkl['train']['labelsAU'].shape[0])
AU_weight = 1.0 / AUoccur_rate
AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
le = SAMM_pkl['train']['labelsAU'].shape[0]
new_aus = np.zeros((le, class_num * class_num))
for j in range(class_num):
	for k in range(class_num):
		new_aus[:,j*class_num+k] = 2 * SAMM_pkl['train']['labelsAU'][:,j] + SAMM_pkl['train']['labelsAU'][:,k]
SAMM_pkl['train']['AU_weight'] = AU_weight
SAMM_pkl['train']['AU_relation'] = new_aus

with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/SAMM/SAMM_subject_independent.pkl', 'wb') as fo:
    pkl.dump(SAMM_pkl, fo)

end_flag = True