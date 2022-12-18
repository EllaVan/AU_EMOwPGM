# 不按subject划分数据集，所有subject的数据混合到一起后再划分数据集
# 且实际上没有区分train和test
import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import os
os.chdir(os.path.dirname(__file__))
import pickle as pkl

import numpy as np
import pandas as pd
import random

from materials.process_priori import cal_interAUPriori

RAF_pkl = {}

phases = ['train', 'test']
raf_path = '/media/data1/Expression/RAF-DB'
list_path_prefix = '../dataset/RAF-DB'
priori_nodes = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust']
data_nodes = ['Happily Surprised', 'Happily Disgusted', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Sadly Disgusted', 
			  'Fearfully Angry', 'Fearfully Surprised', 'Angrily Surprised', 'Angrily Disgusted', 'Disgustedly Surprised']
'''
1: Happily Surprised
2: Happily Disgusted
3: Sadly Fearful
4: Sadly Angry
5: Sadly Surprised
6: Sadly Disgusted
7: Fearfully Angry
8: Fearfully Surprised
9: Angrily Surprised
10: Angrily Disgusted
11: Disgustedly Surprised
'''
classes = len(data_nodes)
RAF_pkl['EMO'] = data_nodes

EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
priori = {'EMO2AU_cpt': EMO2AU_cpt,
		'prob_AU': prob_AU,
		'EMO_img_num': EMO_img_num,
		'AU_cpt': AU_cpt,
		'EMO': EMO,
		'AU': AU}
RAF_pkl['priori'] = priori
RAF_pkl['AU'] = AU[:-2]

NAME_COLUMN = 0
LABEL_COLUMN = 1
weight_EMO_tmp = [0] * len(data_nodes)
for phase in phases:
	temp_img_paths = []
	labelsEMO = []
	RAF_pkl[phase] = {}
	df = pd.read_csv(os.path.join(raf_path, 'compound/EmoLabel/list_patition_label.txt'), sep=' ',
					header=None)
	df = df[df[NAME_COLUMN].str.startswith(phase)]
	df = df.reset_index().iloc[:, 1:]

	for data_node_i, data_node in enumerate(data_nodes):
		row_list = df[df[LABEL_COLUMN] == data_node_i+1].index.tolist()
		df_node = df.iloc[row_list]
		temp_file_names = df_node.iloc[:, NAME_COLUMN].values
		temp_label = data_node_i
		temp_labelEMO = [temp_label] * len(temp_file_names)
		weight_EMO_tmp[temp_label] += len(temp_file_names)

		temp_img_paths = temp_img_paths + temp_file_names.tolist()
		labelsEMO = labelsEMO + temp_labelEMO
	# use raf aligned images for training/testing
	img_paths = []
	weights_EMO = []
	for f_i in range(len(temp_img_paths)):
		f = temp_img_paths[f_i]
		f = f.split(".")[0]
		f = f + "_aligned.jpg"
		path = os.path.join(raf_path, 'compound/Image/aligned', f)
		img_paths.append(path)

		cur_label = labelsEMO[f_i]
		weights_EMO.append(1.0/weight_EMO_tmp[cur_label])
	
	RAF_pkl[phase]['img_path'] = img_paths
	RAF_pkl[phase]['labelsEMO'] = labelsEMO
	labelsAU = np.zeros((len(img_paths), len(AU))) - 1
	RAF_pkl[phase]['labelsAU'] = labelsAU
	if phase == 'train':
		RAF_pkl['train']['EMO_weight'] = weights_EMO

AUoccur_rate = np.ones((1, labelsAU.shape[1]))
AU_weight = 1.0 / AUoccur_rate
AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
RAF_pkl['train']['AU_weight'] = AU_weight

with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/RAF-DB/RAF_compound_subject_independent.pkl', 'wb') as fo:
	pkl.dump(RAF_pkl, fo)
u = 1
