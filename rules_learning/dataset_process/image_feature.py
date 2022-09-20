import os
import argparse
import pickle as pkl
import random

import torch
import torch.utils.data as data

import numpy as np
import pandas as pd

from materials.process_priori import cal_interAUPriori

class BP4DDataSet(data.Dataset):
    def __init__(self, phase=None):
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
        weights_train = []
        datalist_test = {}
        for k, emo in enumerate(EMO):
            random.shuffle(datalist[k])
            random.shuffle(datalist[k])

            train_len = int(0.9 * len(datalist[k]))
            datalist_train += datalist[k][:train_len]
            labellist_train += [k]*train_len
            weights_train += [1.0/train_len] * train_len
            datalist_test_tmp = datalist[k][train_len:]
            datalist_test[k] = tuple(zip([k]*len(datalist_test_tmp), datalist_test_tmp))

        AUlist_train = tuple(zip(labellist_train, datalist_train))
        random.shuffle(list(AUlist_train))
        data_pkl = {}
        data_pkl['EMOCode'] = self.EMO_code_dict
        data_pkl['train'] = AUlist_train
        data_pkl['train_weights'] = weights_train
        data_pkl['test'] = datalist_test
        data_pkl['priori'] = priori
        # with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/BP4D/BP4D.pkl', 'wb') as fo:
        #     pkl.dump(data_pkl, fo)
        end_flag= True


class BP4DDataSet_chuli(data.Dataset):
    def __init__(self, phase=None):
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
                # info_len1 = int(file_info.shape[0] / 11.0 * 5)
                # info_len2 = int(file_info.shape[0] / 11.0 * 6)
                file_info = list(file_info[-5:, :])
                datalist[emo_label] += file_info
        datalist_train = []
        labellist_train = []
        weights_train = []
        datalist_test = {}
        all_test = {}
        for k, emo in enumerate(EMO):

            train_len = int(0.9 * len(datalist[k]))
            datalist_train += datalist[k][:train_len]
            labellist_train += [k]*train_len
            weights_train += [1.0/train_len] * train_len
            datalist_test_tmp = datalist[k][train_len:]
            datalist_test[k] = tuple(zip([k]*len(datalist_test_tmp), datalist_test_tmp))
            all_test[k] = tuple(zip([k]*len(datalist[k]), datalist[k]))

        AUlist_train = tuple(zip(labellist_train, datalist_train))
        data_pkl = {}
        data_pkl['EMOCode'] = self.EMO_code_dict
        data_pkl['train'] = AUlist_train
        data_pkl['train_weights'] = weights_train
        data_pkl['test'] = datalist_test
        data_pkl['priori'] = priori
        data_pkl['all_test'] = all_test
        with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/BP4D/BP4D_chuli2.pkl', 'wb') as fo:
            pkl.dump(data_pkl, fo)
        end_flag= True


class CASMEDataSet(data.Dataset):
    def __init__(self, phase=None):
        EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
        EMO_img_num = [230] * 5

        self.phase = phase
        self.path_info = '/media/data1/wf/AU_EMOwPGM/codes/dataset/CASME/CASME2-coding.xlsx'
        df = pd.read_excel(self.path_info)
        new_col = list(range(df.shape[1]))
        df.columns = new_col
        df = df.drop(2, axis=1)
        df = df.drop(6, axis=1)
        df_data = df.iloc[:, 1]
        df_AU = df.iloc[:, 5]
        df_label = df.iloc[:, 6]
        df = pd.concat([df_data, df_AU, df_label], axis=1)
        new_col = list(range(df.shape[1]))
        df.columns = new_col

        self.nodes = ['happy', 'sad', 'fear', 'surprise', 'disgust']#, 'repression']
        datalist = {}
        EMO = self.nodes
        EMO2index_dict = dict(zip(EMO, range(len(self.nodes)))) #通过名称找到index
        for k, emo in enumerate(EMO):
            datalist[k] = []

        self.priori = {'EMO2AU_cpt': EMO2AU_cpt,
                'prob_AU': prob_AU,
                'EMO_img_num': EMO_img_num,
                'AU_cpt': AU_cpt,
                'EMO': EMO,
                'AU': AU}
                
        for i in range(df.shape[0]):
            item = df.iloc[i, :]
            item_label = item[2]
            if item_label != 'others' and item_label != 'repression' and item_label != 'neutral':
            # if item_label != 'others' and item_label != 'neutral':
                AU_array_tmp = np.zeros((1, 99))
                label = EMO2index_dict[item[2]]
                AU_list = str(item[1]).split('+')
                for au in AU_list:
                    if au[0] == 'R' or au[0] == 'L':
                        au = au[1:]
                    au = int(au)
                    AU_array_tmp[0, au] = 1
                datalist[label].append(list(AU_array_tmp))

        datalist_train = []
        labellist_train = []
        weights_train = []
        datalist_test = {}
        all_test = {}
        for k, emo in enumerate(EMO):
            random.shuffle(datalist[k])
            train_len = int(0.9 * len(datalist[k]))
            datalist_train += datalist[k][:train_len]
            labellist_train += [k]*train_len
            weights_train += [1.0/train_len] * train_len
            datalist_test_tmp = datalist[k][train_len:]
            datalist_test[k] = tuple(zip([k]*len(datalist_test_tmp), datalist_test_tmp))

            all_test[k] = tuple(zip([k]*len(datalist[k]), datalist[k]))

        AUlist_train = tuple(zip(labellist_train, datalist_train))
        random.shuffle(list(AUlist_train))

        data_pkl = {}
        data_pkl['AU'] = AU
        data_pkl['EMO'] = EMO
        data_pkl['train'] = AUlist_train
        data_pkl['train_weights'] = weights_train
        data_pkl['test'] = datalist_test
        data_pkl['priori'] = self.priori
        data_pkl['all_test'] = all_test

        with open('/media/data1/wf/AU_EMOwPGM/codes/dataset/CASME/CAMSE_nR.pkl', 'wb') as fo:
            pkl.dump(data_pkl, fo)
        end_flag= True


if __name__ == '__main__':
    # a = BP4DDataSet()
    b = BP4DDataSet_chuli()
    # c = CASMEDataSet()
    end_flag= True
