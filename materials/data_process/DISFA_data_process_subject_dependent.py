# 按subject划分数据集
import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import os
os.chdir(os.path.dirname(__file__))
import pickle as pkl

import numpy as np
import pandas as pd

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

parts = [['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN009','SN016'],
		 ['SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024'],
		 ['SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']]
# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

# au_idx = AU
au_idx = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
num_AU = len(au_idx)
DISFA_pkl['AU'] = au_idx

part_tmp = {}
for part_i in range(len(parts)):
	part = parts[part_i]
	part_frame_list = []
	part_numpy_list = []
	for fr in part:
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
		part_numpy_list.append(au_label_array)
	part_numpy_list = np.concatenate(part_numpy_list,axis=0)
	part_tmp_name = 'part' +str(part_i+1)
	part_tmp[part_tmp_name] = {}
	part_tmp[part_tmp_name]['img_path'] = part_frame_list
	part_tmp[part_tmp_name]['labelsAU'] = part_numpy_list
	part_tmp[part_tmp_name]['labelsEMO'] = [-1] * len(part_frame_list)

class_num = len(au_idx)
for fold_i in range(3):
	train_part = ['part1', 'part2', 'part3']
	test_part = 'part' + str(3-fold_i)
	train_part.remove(test_part)
	test_fold = 'test_fold' + str(fold_i+1)
	train_fold = 'train_fold' + str(fold_i+1)

	img_path = part_tmp[train_part[0]]['img_path'] + part_tmp[train_part[1]]['img_path']
	labelsEMO = part_tmp[train_part[0]]['labelsEMO'] + part_tmp[train_part[1]]['labelsEMO']
	labelsAU = np.concatenate((part_tmp[train_part[0]]['labelsAU'], 
							   part_tmp[train_part[1]]['labelsAU']), axis=0)

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
			
	DISFA_pkl[train_fold] = {}
	DISFA_pkl[train_fold]['img_path'] = img_path
	DISFA_pkl[train_fold]['labelsAU'] = labelsAU
	DISFA_pkl[train_fold]['labelsEMO'] = labelsEMO
	DISFA_pkl[train_fold]['AU_weight'] = AU_weight
	DISFA_pkl[train_fold]['AU_relation'] = new_aus

	DISFA_pkl[test_fold] = {}
	DISFA_pkl[test_fold]['img_path'] = part_tmp[test_part]['img_path']
	DISFA_pkl[test_fold]['labelsAU'] = part_tmp[test_part]['labelsAU']
	DISFA_pkl[test_fold]['labelsEMO'] = part_tmp[test_part]['labelsEMO']

with open('/media/data1/wf/AU_EMOwPGM/codes/dataset//DISFA/DISFA_subject_dependent.pkl', 'wb') as fo:
	pkl.dump(DISFA_pkl, fo)
u = 1


