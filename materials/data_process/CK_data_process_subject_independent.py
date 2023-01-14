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

CK_pkl = {}

phases = ['train', 'test']
CK_path = '/media/data1/Expression/CK+/processed'
list_path_prefix = '../dataset/CK+'
priori_nodes = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust']
data_nodes = ['happy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
classes = len(data_nodes)
CK_pkl['EMO'] = data_nodes

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
priori = {'EMO2AU_cpt': EMO2AU_cpt,
		'prob_AU': prob_AU,
		'EMO_img_num': EMO_img_num,
		'AU_cpt': AU_cpt,
		'EMO': EMO,
		'AU': AU}
CK_pkl['priori'] = priori
priori_rules = get_priori()
CK_pkl['priori_rules'] = priori_rules
CK_pkl['AU'] = AU[:-2]

split = [0.9, 0.1]

labels = []
files = []
EMO_weight = []
for label, node in enumerate(data_nodes):
	file_node = os.path.join(CK_path, node)
	file_names = []
	img_files = []
	for home, dirs1, dirs2 in os.walk(file_node):
		for tmp_file in dirs1:
			img_files.append(os.path.join(home, tmp_file))
		break
	for img_file_i in range(len(img_files)):
		for home, dirs1, dirs2 in os.walk(img_files[img_file_i]):
			for tmp_file in dirs2:
				file_names.append(os.path.join(home, tmp_file))
			break
	files.append(file_names)
	EMO_weight.append(1.0/len(file_names))
for phase in phases:
	CK_pkl[phase] = {}
	paths = []
	labels = []
	if phase == 'train':
		paths = []
		labels = []
		for class_i in range(len(data_nodes)):
			file_names = files[class_i]
			imgs_num = len(file_names)
			tr_num = int(imgs_num * split[0])
			for tr_img_i in range(tr_num):
				paths.append(file_names[tr_img_i])
				labels.append(class_i)
	elif phase == 'test':
		paths = []
		labels = []
		for class_i in range(len(data_nodes)):
			file_names = files[class_i]
			imgs_num = len(file_names)
			tr_num = int(imgs_num * split[0])
			for te_img_i in range(tr_num, imgs_num):
				paths.append(file_names[te_img_i])
				labels.append(class_i)

	
	
	CK_pkl[phase]['img_path'] = paths
	CK_pkl[phase]['labelsEMO'] = labels
	labelsAU = np.zeros((len(paths), len(AU))) - 1
	CK_pkl[phase]['labelsAU'] = labelsAU
	if phase == 'train':
		CK_pkl['train']['EMO_weight'] = EMO_weight

AUoccur_rate = np.ones((1, labelsAU.shape[1]))
AU_weight = 1.0 / AUoccur_rate
AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
CK_pkl['train']['AU_weight'] = AU_weight

with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/CK+/CK_subject_independent.pkl', 'wb') as fo:
	pkl.dump(CK_pkl, fo)
u = 1
