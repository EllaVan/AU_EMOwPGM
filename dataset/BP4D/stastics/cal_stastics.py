'''
EMO2AU的顺序就是BP4D原task的顺序: happy, sad, surprise, fear, anger disgust
顺序和priori不一致
'''

import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

import os
import csv
import pickle as pkl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from materials.process_priori import cal_interAUPriori

def interAU():
    path_info = '/media/database/data2/Expression/BP4D/AUCoding/AU_OCC'
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    AU = AU[:-2]
    num_AU = len(AU)
    stat_ij = np.zeros((num_AU, num_AU))
    stat_au = [0] * num_AU
    train_size = 0

    file_list = os.listdir(path_info)
    for file in file_list:
        # 当前数据的表情标签由文件标题中的任务决定
        task = file.split('.')[0].split('_')[-1]
        if task != 'T4' and task != 'T6':
            cur_info_path = os.path.join(path_info, file)
            file_info = np.array(pd.read_csv(cur_info_path)) # 文件中AU的发生情况从第二列起，第一列为frame编码
            for i in range(file_info.shape[0]):
                cur_item = file_info[i, AU]
                cur_item = np.where(cur_item != 9, cur_item, 0)
                mask_info = np.array(np.nonzero(cur_item)).squeeze(0).tolist()
                if len(mask_info) != 0:
                    train_size += 1
                    for j1 in mask_info:
                        for j2 in mask_info:
                            stat_ij[j1][j2] += 1
                        stat_au[j1] += 1

    for i in range(num_AU):
        stat_ij[:, i] = stat_ij[:, i] / stat_au[i]  # stat_ij[i, j] = P(AUi | AUj)
        stat_au[i] = stat_au[i] / train_size
    
    return stat_ij, stat_au, train_size


def dataset_EMO2AU():
    path_info = '/media/database/data2/Expression/BP4D/AUCoding/AU_OCC'
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    AU = AU[:-2]
    num_AU = len(AU)

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
    del EMO_code_dict['T4']
    del EMO_code_dict['T6']
    tasks = list(EMO_code_dict.keys())
    EMO = list(EMO_code_dict.values())
    num_tasks = len(tasks)
    tasks2index = dict(zip(tasks, range(len(tasks))))
    stat = {}
    stat['EMO_size'] = [0] * len(EMO)
    for i in range(len(EMO_code_dict)):
        AU_cnt = dict(zip(list(range(0, num_AU)), [0]*num_AU))
        stat[tasks[i]] = AU_cnt

    file_list = os.listdir(path_info)
    for file in file_list:
        # 当前数据的表情标签由文件标题中的任务决定
        task = file.split('.')[0].split('_')[-1]
        if task != 'T4' and task != 'T6':
            task_index = tasks2index[task]
            cur_info_path = os.path.join(path_info, file)
            file_info = np.array(pd.read_csv(cur_info_path)) # 文件中AU的发生情况从第二列起，第一列为frame编码
            for i in range(file_info.shape[0]):
                cur_item = file_info[i, AU]
                cur_item = np.where(cur_item != 9, cur_item, 0)
                mask_info = np.array(np.nonzero(cur_item)).squeeze(0).tolist()
                if len(mask_info) != 0:
                    stat['EMO_size'][task_index] += 1
                    for j1 in mask_info:
                        stat[task][j1] += 1

    EMO2AU = np.zeros((num_tasks, num_AU))
    for i in range(num_tasks):
        for j in range(num_AU):
            a = tasks[i]
            EMO2AU[i][j] = stat[tasks[i]][j] / stat['EMO_size'][tasks2index[a]]

    img_num = stat['EMO_size']
    return EMO, AU, EMO2AU, img_num


def dataset_sort_EMO2AU():
    path_info = '/media/data1/wf/AU_EMOwPGM/codes/save/stastics/stastics_EMO2AU>=3.csv'
    EMO2AU = np.array(pd.read_csv(path_info, header=None))
    a = np.argsort(-EMO2AU[0, :])
    print('AU occ of happy (from high to low): ', end='')
    for j in range(5):
        if j != 4:
            print('AU'+str(a[j]), end=', ')
        else:
            print('AU'+str(a[j]))
    a = np.argsort(-EMO2AU[1, :])
    print('AU occ of sad (from high to low): ', end='')
    for j in range(5):
        if j != 4:
            print('AU'+str(a[j]), end=', ')
        else:
            print('AU'+str(a[j]))
    a = np.argsort(-EMO2AU[4, :])
    print('AU occ of fear (from high to low): ', end='')
    for j in range(5):
        if j != 4:
            print('AU'+str(a[j]), end=', ')
        else:
            print('AU'+str(a[j]))
    a = np.argsort(-EMO2AU[6, :])
    print('AU occ of anger (from high to low): ', end='')
    for j in range(5):
        if j != 4:
            print('AU'+str(a[j]), end=', ')
        else:
            print('AU'+str(a[j]))
    a = np.argsort(-EMO2AU[2, :])
    print('AU occ of surprise (from high to low): ', end='')
    for j in range(5):
        if j != 4:
            print('AU'+str(a[j]), end=', ')
        else:
            print('AU'+str(a[j]))
    a = np.argsort(-EMO2AU[7, :])
    print('AU occ of disgust (from high to low): ', end='')
    for j in range(5):
        if j != 4:
            print('AU'+str(a[j]), end=', ')
        else:
            print('AU'+str(a[j]))

    for i in range(EMO2AU.shape[0]):
        print(np.argsort(-EMO2AU[i, :]))


if __name__ == '__main__':
    # stat_ij, stat_au, train_size = interAU()
    EMO, AU, EMO2AU, img_num = dataset_EMO2AU()
    # data_pkl = {}
    # data_pkl['EMO'] = EMO
    # data_pkl['AU'] = AU
    # data_pkl['sta_AU_cpt'] = stat_ij
    # data_pkl['sta_prob_AU'] = stat_au
    # data_pkl['sta_EMO2AU_cpt'] = EMO2AU
    # data_pkl['img_num'] = img_num
    # with open('/media/data1/wf/AU_EMOwPGM/codes/save/BP4D/stastics/stat_basicEMO.pkl', 'wb') as fo:
    #     pkl.dump(data_pkl, fo)
    end_flag = True