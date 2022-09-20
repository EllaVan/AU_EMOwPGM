# 不按subject划分数据集，所有subject的数据混合到一起后再划分数据集
import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import os
os.chdir(os.path.dirname(__file__))
import pickle as pkl

import numpy as np
import pandas as pd
import random

from materials.process_priori import cal_interAUPriori

AffectNet_pkl = {}

phases = [('train', 'training'), ('test', 'validation')]
dataset_path = '/media/data1/Expression/AffectNet'
list_path_prefix = '../dataset/AffectNet'
priori_nodes = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust']
data_nodes = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger',
			  # 'contempt'
			 ]
classes = len(data_nodes)
AffectNet_pkl['EMO'] = priori_nodes

EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
priori = {'EMO2AU_cpt': EMO2AU_cpt,
		'prob_AU': prob_AU,
		'EMO_img_num': EMO_img_num,
		'AU_cpt': AU_cpt,
		'EMO': EMO,
		'AU': AU}
AffectNet_pkl['priori'] = priori
AffectNet_pkl['AU'] = AU[:-2]

NAME_COLUMN = 0
LABEL_COLUMN = 1
weight_EMO_tmp = [0] * len(data_nodes[1:])
for phase, label_usage in phases:
	labelsEMO = []
	AffectNet_pkl[phase] = {}
	df = pd.read_csv(os.path.join(dataset_path, 'label', label_usage+'.csv'), low_memory=False,
					header=None).drop(0)
	df_name = df.iloc[:, 0]
	df_usage = df.iloc[:, -1]
	df_label = df.iloc[:, 6]
	df = pd.concat([df_name, df_label, df_usage], axis=1)
	new_col = [0, 1, 2]
	df.columns = new_col
	NAME_COLUMN = 0
	LABEL_COLUMN = 1
	df[LABEL_COLUMN] = df[LABEL_COLUMN].astype('int')
	row_list7 = df[df[LABEL_COLUMN] == 7].index.tolist()  # get the row of neutral and delete them
	df = df.drop(row_list7)
	row_list8 = df[df[LABEL_COLUMN] == 8].index.tolist()
	df = df.drop(row_list8)
	row_list9 = df[df[LABEL_COLUMN] == 9].index.tolist()
	df = df.drop(row_list9)
	row_list10 = df[df[LABEL_COLUMN] == 10].index.tolist()
	df = df.drop(row_list10)
	
	usage_column = 2
	df_temp = []
	for data_node_i, data_node in enumerate(data_nodes[1:]):
		dfi = df[df[LABEL_COLUMN] == data_node_i]
		df_temp.append(dfi)

	file_names = []
	labelsEMO = []
	EMO_weights = []
	for tr_i in range(len(df_temp)):
		names = df_temp[tr_i].iloc[:, NAME_COLUMN].values
		file_names.append(names)
		emo_name = data_nodes[tr_i+1]
		temp_label = priori_nodes.index(emo_name)
		label = np.array([temp_label] * names.shape[0])
		labelsEMO.append(label)
		EMO_weight = 1.0 / names.shape[0]
		EMO_weights.append([EMO_weight] * names.shape[0])
	labelsEMO = np.hstack(labelsEMO)
	file_names = np.hstack(file_names)
	EMO_weights = np.hstack(EMO_weights)
	file_paths = []
	for f in file_names:
		f = f.split(".")[0]
		f = f + ".jpg"
		path = os.path.join(dataset_path, 'data/AffectNet', f)
		file_paths.append(path)
	AffectNet_pkl[phase]['img_path'] = file_paths
	AffectNet_pkl[phase]['labelsEMO'] = labelsEMO
	labelsAU = np.zeros((len(file_paths), len(AU))) - 1
	AffectNet_pkl[phase]['labelsAU'] = labelsAU
	if phase == 'train':
		AffectNet_pkl['train']['EMO_weight'] = EMO_weights
		AUoccur_rate = np.ones((1, labelsAU.shape[1]))
		AU_weight = 1.0 / AUoccur_rate
		AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
		AffectNet_pkl['train']['AU_weight'] = AU_weight

with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/AffectNet/AffectNet_subject_independent.pkl', 'wb') as fo:
	pkl.dump(AffectNet_pkl, fo)
u = 1
