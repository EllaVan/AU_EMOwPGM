import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

import argparse
import datetime
from errno import EMULTIHOP
import pytz
import os
import shutil
import pickle as pkl
import random

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.utils import get_example_model

from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs, interAUs
import rules_learning.utils as utils

def interAU():
    path_info = '/media/database/data2/Expression/BP4D/AUCoding/AU_OCC'

    train_loader, test_loader = utils.getDatasetInfo('BP4D')
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
    num_AU = len(AU)
    stat_ij = np.zeros((num_AU, num_AU))
    stat_au = [0]*num_AU

    for idx, (cur_item, emo_label, index) in enumerate(train_loader, 1):
        occ_au = []
        if len(cur_item.shape) > 2:
            cur_item = cur_item.squeeze(0)
        cur_item = np.where(cur_item != 9, cur_item, 0)
        
        mask_info = np.array(np.nonzero(cur_item)).squeeze(0).tolist()
        if len(mask_info) > 1:
            for j1 in mask_info:
                for j2 in mask_info:
                    stat_ij[j1+1][j2+1] += 1
                stat_au[j1+1] += 1

    for i in range(1, stat_ij.shape[0]):
        stat_ij[:, i] = stat_ij[:, i] / stat_au[i]  # stat_ij[i, j] = P(AUi | AUj)

    with open('/media/data1/wf/AU_EMOwPGM/codes/save/stastics/au_ij.csv', 'w', encoding='UTF8', newline='') as f: 
        csv_writer = csv.writer(f)
        csv_writer.writerows(stat_ij)

    


def dataset_EMO2AU():
    path_info = '/media/database/data2/Expression/BP4D/AUCoding/AU_OCC'
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
    tasks = list(EMO_code_dict.keys())
    EMO2index = dict(zip(tasks, range(len(tasks))))
    stat = {}
    for i in range(len(EMO_code_dict)):
        AU_cnt = dict(zip(list(range(0, 28)), [0]*28))
        stat[tasks[i]] = AU_cnt

    file_list = os.listdir(path_info)
    for file in file_list:
        # 当前数据的表情标签由文件标题中的任务决定
        task = file.split('.')[0].split('_')[-1]

        cur_info_path = os.path.join(path_info, file)
        file_info = np.array(pd.read_csv(cur_info_path)) # 文件中AU的发生情况从第二列起，第一列为frame编码
        for i in range(file_info.shape[0]):
            cur_item = file_info[i, 1:27]
            cur_item = np.where(cur_item != 9, cur_item, 0)
            mask_info = np.array(np.nonzero(cur_item)).squeeze(0).tolist()
            if len(mask_info) > 1:
                for j1 in mask_info:
                    stat[task][j1] += 1

    EMO2AU = np.zeros((len(tasks), 27))
    for i in range(EMO2AU.shape[0]):
        for j in range(1, EMO2AU.shape[1]):
            EMO2AU[i][j] = stat[tasks[i]][j] / stat[tasks[i]][0]

    df = pd.DataFrame(EMO2AU, index=list(EMO_code_dict.values()), 
                        columns=['AU'+str(i) for i in list(range(EMO2AU.shape[1]))])
    f, ax = plt.subplots(figsize=(26,5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt='.3f', ax = ax)
    plt.savefig("a.jpg", dpi=500)
    plt.show()


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

    end_flag = True