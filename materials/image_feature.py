import os
import argparse
import pickle as pkl
import random

import torch
import torch.utils.data as data

import numpy as np
import pandas as pd

from process_priori import cal_interAUPriori


EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
index2EMO_dict = dict(zip(range(len(EMO)), EMO))
EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index
AU2index_dict = dict(zip(list(map(int, AU)), range(len(AU))))
index2AU_dict = dict(zip(range(len(AU)), list(map(int, AU))))
priori = {'EMO2AU_cpt': EMO2AU_cpt,
        'prob_AU': prob_AU,
        'EMO_img_num': EMO_img_num,
        'AU_cpt': AU_cpt,
        'EMO': EMO,
        'AU': AU}


class AUList(object):
    def __init__(self, phase=None):

        pass


class BP4DDataSet(data.Dataset):
    def __init__(self, phase=None):
        self.phase = phase
        self.path_info = '/media/database/data2/Expression/BP4D/AUCoding/AU_OCC'
        self.EMO_code_dict = {
            'T1': 'happy',
            'T2': 'sad',
            'T3': 'surprise',
            'T4': 'embarrassment',
            'T5': 'fear',
            'T6': 'physical pain',
            'T7': 'anger',
            'T8': 'disgust',
        }
        datalist = {}
        for k, emo in enumerate(EMO):
            datalist[k] = []

        file_list = os.listdir(self.path_info)
        for file in file_list:
            # 当前数据的表情标签由文件标题中的任务决定
            task = file.split('.')[0].split('_')[-1]
            if task != 'T4' and task != 'T6': # 暂时不考虑基本表情以外的任务
                emo_name = self.EMO_code_dict[task] 
                emo_label = EMO2index_dict[emo_name]
                
                cur_info_path = os.path.join(self.path_info, file)
                file_info = np.array(pd.read_csv(cur_info_path)) # 文件中AU的发生情况从第二列起，第一列为frame编码
                # datalist[emo_label].append(list(file_info))
                datalist[emo_label] += list(file_info)
        datalist_train = []
        labellist_train = []
        datalist_test = {}
        for k, emo in enumerate(EMO):
            random.shuffle(datalist[k])
            random.shuffle(datalist[k])

            train_len = int(0.9 * len(datalist[k]))
            datalist_train += datalist[k][:train_len]
            labellist_train += [k]*train_len
            datalist_test_tmp = datalist[k][train_len:]
            datalist_test[k] = tuple(zip([k]*len(datalist_test_tmp), datalist_test_tmp))

        AUlist_train = tuple(zip(labellist_train, datalist_train))
        random.shuffle(list(AUlist_train))
        data_pkl = {}
        data_pkl['EMOCode'] = self.EMO_code_dict
        data_pkl['train'] = AUlist_train
        data_pkl['test'] = datalist_test
        data_pkl['priori'] = priori
        with open('/media/database/data4/wf/AU_EMOwPGM/codes/dataset/BP4D2.pkl', 'wb') as fo:
            pkl.dump(data_pkl, fo)
        end_flag= True


if __name__ == '__main__':
    a = BP4DDataSet()
