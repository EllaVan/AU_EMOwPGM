# 按subject划分数据集
import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import os
os.chdir(os.path.dirname(__file__))
import pickle as pkl

import numpy as np
import pandas as pd

from materials.process_priori import cal_interAUPriori

BP4D_pkl = {}

label_folder = '/media/data1/Expression/BP4D/AUCoding/AU_OCC'
img_root_path = '/media/data1/Expression/BP4D/BP4D_MTCNN'
list_path_prefix = '../dataset/BP4D'
EMO_code_dict = {
				'T1': 'happy',
				'T2': 'sad',
				'T3': 'surprise',
				'T4': 'embarrassment',
				'T5': 'fear',
				'T6': 'physical pain',
				'T7': 'anger',
				'T8': 'disgust',
				}
tasks = ['T1', 'T2', 'T3', 'T5', 'T7', 'T8'] # 暂时不考虑基本表情以外的任务

EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
index2EMO_dict = dict(zip(range(len(EMO)), EMO))
EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index
AU2index_dict = dict(zip(list(map(int, AU)), range(len(AU))))
index2AU_dict = dict(zip(range(len(AU)), list(map(int, AU))))
priori = {'EMO2AU_cpt': EMO2AU_cpt,
		'prob_AU': prob_AU,
		'EMO_img_num': EMO_img_num,
		'AU_cpt': AU_cpt,
		'EMO': EMO, # happy, sad, fear, anger, surprise, disgust
		'AU': AU}
BP4D_pkl['priori'] = priori

BP4D_Sequence_split = [['F001','M007','F018','F008','M004','F010','F009','M012','M001','F016','M014','F023','M008'], # 'F001','M007','F018','F008','F002','M004','F010','F009','M012','M001','F016','M014','F023','M008'
					   ['M011','F003','M010','M002','F005','F022','M018','M017','F013','M016','F020','F011','M013','M005'],
					   ['F007','F015','F006','F019','M006','M009','F012','M003','F004','F021','F017','M015','F014']]
# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

BP4D_pkl['EMO'] = EMO
AU = AU[:-2]
BP4D_pkl['AU'] = AU
def get_AUlabels(seq, task, path):
	path_label = os.path.join(path, '{sequence}_{task}.csv'.format(sequence=seq, task=task))
	usecols = [str(i) for i in AU] + ['0']
	df = pd.read_csv(path_label, header=0, index_col=0, usecols=usecols)
	frames = [str(item) for item in list(df.index.values)]
	frames_path = ['{}/{}/{}'.format(seq, task, item) for item in frames]
	labels_AU = df.values
	# 返回的frames是list，值是排好序的int变量，指示对应的帧。labels_AU是N*12的np.ndarray，对应AU标签
	return frames_path, labels_AU

part_tmp = {}
for split_i in range(len(BP4D_Sequence_split)):
	sequences = BP4D_Sequence_split[split_i]
	frames = None
	labelsAU = None
	labelsEMO = None
	for seq in sequences:
		for t in tasks:
			temp_frames, temp_labelsAU = get_AUlabels(seq, t, label_folder)
			emo_name = EMO_code_dict[t]
			temp_labelsEMO = EMO2index_dict[emo_name]
			temp_labelsEMO = [temp_labelsEMO] * len(temp_frames)
			if frames is None:
				labelsAU = temp_labelsAU
				frames = temp_frames  # str list
				labelsEMO = temp_labelsEMO
			else:
				labelsAU = np.concatenate((labelsAU, temp_labelsAU), axis=0)  # np.ndarray
				frames = frames + temp_frames  # str list
				labelsEMO = labelsEMO + temp_labelsEMO

	for frame_i in range(len(frames)):
		frames[frame_i] = frames[frame_i] + '.jpg'
		img = frames[frame_i]

		zfill_flag = 0
		img_split = img.split('/')
		pad_path = img_split[0]
		img_path = os.path.join(img_root_path, pad_path, img)
		if os.path.exists(img_path):
			pass
		else:
			img_idx = img_split[-1].split('.')[0]
			while os.path.exists(img_path) is False and zfill_flag < 3:
				zfill_flag += 1
				img_idx = img_idx.zfill(len(img_idx)+1)
				a = img.split('/')[-1]
				img = os.path.join(img.strip(a), img_idx+'.'+img_split[-1].split('.')[-1])
				img_path = os.path.join(img_root_path, pad_path, img)
		frames[frame_i] = img_path
	
	part_tmp_name = 'part' +str(split_i+1)
	part_tmp[part_tmp_name] = {}
	part_tmp[part_tmp_name]['img_path'] = frames
	part_tmp[part_tmp_name]['labelsAU'] = labelsAU
	part_tmp[part_tmp_name]['labelsEMO'] = labelsEMO

class_num = len(AU)
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
			
	BP4D_pkl[train_fold] = {}
	BP4D_pkl[train_fold]['img_path'] = img_path
	BP4D_pkl[train_fold]['labelsAU'] = labelsAU
	BP4D_pkl[train_fold]['labelsEMO'] = labelsEMO
	BP4D_pkl[train_fold]['AU_weight'] = AU_weight
	BP4D_pkl[train_fold]['AU_relation'] = new_aus

	BP4D_pkl[test_fold] = {}
	BP4D_pkl[test_fold]['img_path'] = part_tmp[test_part]['img_path']
	BP4D_pkl[test_fold]['labelsAU'] = part_tmp[test_part]['labelsAU']
	BP4D_pkl[test_fold]['labelsEMO'] = part_tmp[test_part]['labelsEMO']

with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/BP4D/BP4D_subject_dependent.pkl', 'wb') as fo:
	pkl.dump(BP4D_pkl, fo)

end_flag = True