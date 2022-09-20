# 不按subject划分数据集，所有subject的数据混合到一起后再划分数据集
# 每个subject的的每个任务的数据都按照7:3划分
import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import os
os.chdir(os.path.dirname(__file__))
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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

BP4D_Sequence_split = ['F001','M007','F018','F008','M004','F010','F009','M012','M001','F016','M014','F023','M008', # 'F001','M007','F018','F008','F002','M004','F010','F009','M012','M001','F016','M014','F023','M008'
					   'M011','F003','M010','M002','F005','F022','M018','M017','F013','M016','F020','F011','M013','M005',
					   'F007','F015','F006','F019','M006','M009','F012','M003','F004','F021','F017','M015','F014']

BP4D_pkl['EMO'] = EMO
AU = AU[:-2]
BP4D_pkl['AU'] = AU
def get_AUlabels(seq, task, path):
	path_label = os.path.join(path, '{sequence}_{task}.csv'.format(sequence=seq, task=task))
	usecols = [str(i) for i in AU] + ['0']
	df = pd.read_csv(path_label, header=0, index_col=0, usecols=usecols)
	df = shuffle(df)
	frames = [str(item) for item in list(df.index.values)]
	frames_path = ['{}/{}/{}'.format(seq, task, item) for item in frames]
	labels_AU = df.values
	# 返回的frames是list，值是排好序的int变量，指示对应的帧。labels_AU是N*12的np.ndarray，对应AU标签
	return frames_path, labels_AU

train_ratio = 0.7
test_ratio = 1 - train_ratio
train_frames = None
test_frames = None
train_labelsAU = None
test_labelsAU = None
train_labelsEMO = None
test_labelsEMO = None
weight_EMO_tmp = [0] * len(tasks)
for seq in BP4D_Sequence_split:
	for t in tasks:
		temp_frames, temp_labelsAU = get_AUlabels(seq, t, label_folder)
		emo_name = EMO_code_dict[t]
		temp_labelsEMO = EMO2index_dict[emo_name]

		temp_all_len = len(temp_frames)
		temp_train_len = int(train_ratio * temp_all_len)
		temp_test_len = temp_all_len - temp_train_len
		weight_EMO_tmp[temp_labelsEMO] += temp_train_len

		temp_train_frames = temp_frames[:temp_train_len]
		temp_test_frames = temp_frames[temp_train_len:]
		temp_train_labelsAU = temp_labelsAU[:temp_train_len]
		temp_test_labelsAU = temp_labelsAU[temp_train_len:]
		temp_train_labelsEMO = [temp_labelsEMO] * temp_train_len
		temp_test_labelsEMO = [temp_labelsEMO] * temp_test_len
		
		if train_frames is None:
			train_frames = temp_train_frames
			test_frames = temp_test_frames
			train_labelsAU = temp_train_labelsAU
			test_labelsAU = temp_test_labelsAU
			train_labelsEMO = temp_train_labelsEMO
			test_labelsEMO = temp_test_labelsEMO
		else:
			train_frames = train_frames + temp_train_frames
			test_frames = test_frames + temp_test_frames
			train_labelsAU = np.concatenate((train_labelsAU, temp_train_labelsAU), axis=0)  # np.ndarray
			test_labelsAU = np.concatenate((test_labelsAU, temp_test_labelsAU), axis=0)  # np.ndarray
			train_labelsEMO = train_labelsEMO + temp_train_labelsEMO
			test_labelsEMO = test_labelsEMO + temp_test_labelsEMO

weights_EMO = []
for frame_i in range(len(train_frames)):
	train_frames[frame_i] = train_frames[frame_i] + '.jpg'
	img = train_frames[frame_i]

	cur_label = train_labelsEMO[frame_i]
	weights_EMO.append(1.0/weight_EMO_tmp[cur_label])

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
	train_frames[frame_i] = img_path

for frame_i in range(len(test_frames)):
	test_frames[frame_i] = test_frames[frame_i] + '.jpg'
	img = test_frames[frame_i]

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
	test_frames[frame_i] = img_path
	
BP4D_pkl['train'] = {}
BP4D_pkl['train']['img_path'] = train_frames
BP4D_pkl['train']['labelsAU'] = train_labelsAU
BP4D_pkl['train']['labelsEMO'] = train_labelsEMO
BP4D_pkl['train']['EMO_weight'] = weights_EMO
BP4D_pkl['test'] = {}
BP4D_pkl['test']['img_path'] = test_frames
BP4D_pkl['test']['labelsAU'] = test_labelsAU
BP4D_pkl['test']['labelsEMO'] = test_labelsEMO

class_num = len(AU)
AUoccur_rate = np.zeros((1, train_labelsAU.shape[1]))
for i in range(train_labelsAU.shape[1]):
	AUoccur_rate[0, i] = sum(train_labelsAU[:,i]>0) / float(train_labelsAU.shape[0])
AU_weight = 1.0 / AUoccur_rate
AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
le = train_labelsAU.shape[0]
new_aus = np.zeros((le, class_num * class_num))
for j in range(class_num):
	for k in range(class_num):
		new_aus[:,j*class_num+k] = 2 * train_labelsAU[:,j] + train_labelsAU[:,k]
BP4D_pkl['train']['AU_weight'] = AU_weight
BP4D_pkl['train']['AU_relation'] = new_aus

with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/BP4D/BP4D_subject_independent.pkl', 'wb') as fo:
	pkl.dump(BP4D_pkl, fo)

end_flag = True