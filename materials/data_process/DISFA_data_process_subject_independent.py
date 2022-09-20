# 不按subject划分数据集，所有subject的数据混合到一起后再划分数据集
# 每个subject的的每个任务的数据都按照7:3划分
import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import os
os.chdir(os.path.dirname(__file__))
import pickle as pkl

import numpy as np
import pandas as pd
import random

from materials.process_priori import cal_interAUPriori

# with open('/media/data1/hx/code/CVPR/data/DISFA_MTCNN_pkl/val9.pkl', 'rb') as fo:
#     pkl_file = pkl.load(fo)

DISFA_pkl = {}

label_path = '/media/data1/Expression/DISFA/DISFA_init/ActionUnit_Labels'
img_root_path = '/media/data1/Expression/DISFA/DISFA_MTCNN/Video_RightCamera'
list_path_prefix = '../data/DISFA'

EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
priori = {'EMO2AU_cpt': EMO2AU_cpt,
		'prob_AU': prob_AU,
		'EMO_img_num': EMO_img_num,
		'AU_cpt': AU_cpt,
		'EMO': EMO,
		'AU': AU}
DISFA_pkl['priori'] = priori
DISFA_pkl['EMO'] = EMO

parts = ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN009','SN016',
		 'SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024',
		 'SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']

# au_idx = AU
au_idx = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
num_AU = len(au_idx)
DISFA_pkl['AU'] = au_idx

train_ratio = 0.7
test_ratio = 1 - train_ratio
train_part_frame_list = []
train_part_numpy_list = []
train_part_EMO_list = []
test_part_frame_list = []
test_part_numpy_list = []
test_part_EMO_list = []
for fr in parts:
	part_frame_list = []
	fr_path = os.path.join(label_path,fr)
	au1_path = os.path.join(fr_path,fr+'_au1.txt')
	with open(au1_path, 'r') as label:
		total_frame = len(label.readlines())
	au_label_array = np.zeros((total_frame, num_AU),dtype=int)
	for ai, au in enumerate(au_idx):
		AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
		if not os.path.isfile(AULabel_path):
			continue
		print("--Checking AU:" + str(au) + " ...")
		with open(AULabel_path, 'r') as label:
			for t, lines in enumerate(label.readlines()):
				frameIdx, AUIntensity = lines.split(',')
				frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
				if AUIntensity >= 2:
					AUIntensity = 1
				else:
					AUIntensity = 0
				au_label_array[t,ai] = AUIntensity
	for i in range(total_frame):
		# frame_subject_name = 'right_' + fr
		frame_img_name = 'right_' + fr + '/' + str(i+1) + '.jpg'
		img = os.path.join(img_root_path, frame_img_name)
		part_frame_list.append(img)

	fr_all_len = len(part_frame_list)
	train_len = int(train_ratio * fr_all_len)
	test_len = fr_all_len - train_len

	temp_AU_list = list(au_label_array)
	temp_info = list(zip(part_frame_list, temp_AU_list))
	random.shuffle(temp_info)

	for i in range(train_len):
		train_part_frame_list.append(temp_info[i][0])
		train_part_numpy_list.append(temp_info[i][1].reshape(1, -1))
	for j in range(train_len, fr_all_len):
		test_part_frame_list.append(temp_info[j][0])
		test_part_numpy_list.append(temp_info[j][1].reshape(1, -1))

train_part_numpy_list = np.concatenate(train_part_numpy_list,axis=0)
test_part_numpy_list = np.concatenate(test_part_numpy_list,axis=0)

DISFA_pkl['train'] = {}
DISFA_pkl['train']['img_path'] = train_part_frame_list
DISFA_pkl['train']['labelsAU'] = train_part_numpy_list
DISFA_pkl['train']['labelsEMO'] = [-1]*len(train_part_frame_list)
DISFA_pkl['test'] = {}
DISFA_pkl['test']['img_path'] = test_part_frame_list
DISFA_pkl['test']['labelsAU'] = test_part_numpy_list
DISFA_pkl['test']['labelsEMO'] = [-1]*len(test_part_frame_list)

class_num = len(au_idx)
labelsAU = train_part_numpy_list
AUoccur_rate = np.zeros((1, labelsAU.shape[1]))
for i in range(labelsAU.shape[1]):
	AUoccur_rate[0, i] = sum(labelsAU[:,i]>0) / float(labelsAU.shape[0])
AU_weight = 1.0 / AUoccur_rate
AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]

le = labelsAU.shape[0]
new_aus = np.zeros((le, class_num * class_num))
for j in range(class_num):
	for k in range(class_num):
		new_aus[:,j*class_num+k] = 2 * labelsAU[:,j] + labelsAU[:,k]

DISFA_pkl['train']['AU_weight'] = AU_weight
DISFA_pkl['train']['AU_relation'] = new_aus

with open('/media/data1/wf/AU_EMOwPGM/codes/dataset//DISFA/DISFA_subject_independent.pkl', 'wb') as fo:
	pkl.dump(DISFA_pkl, fo)
u = 1


