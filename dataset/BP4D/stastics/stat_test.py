import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch

from materials.process_priori import cal_interAUPriori
from models.AU_EMO_BP import UpdateGraph
import rules_learning.utils as utils


def interAU():
    _, _, _, _, EMO, AU = cal_interAUPriori()
    del(AU[-1])
    del(AU[-1])
    path_info = '/media/database/data2/Expression/BP4D/AUCoding/AU_OCC'
    stat_ij = np.zeros((len(AU), len(AU)))
    stat_au = [0]*len(AU)
    stat_size = 0

    file_list = os.listdir(path_info)
    for file in file_list:
        cur_info_path = os.path.join(path_info, file)
        file_info = np.array(pd.read_csv(cur_info_path)) # 文件中AU的发生情况从第二列起，第一列为frame编码
        for i in range(file_info.shape[0]):
            cur_item = file_info[i, AU]
            cur_item = np.where(cur_item != 9, cur_item, 0)
            mask_info = np.array(np.nonzero(cur_item)).squeeze(0).tolist()
            if len(mask_info) >= 1:
                stat_size += 1
                for j1 in mask_info:
                    for j2 in mask_info:
                        stat_ij[j1][j2] += 1
                    stat_au[j1] += 1

    for i in range(stat_ij.shape[0]):
        stat_ij[:, i] = stat_ij[:, i] / stat_au[i]  # stat_ij[i, j] = P(AUi | AUj)
    AU_cpt = stat_ij
    prob_AU = np.array(stat_au) / stat_size
    return AU_cpt, prob_AU


def dataset_EMO2AU():
    _, _, _, _, EMO, AU = cal_interAUPriori()
    path_info = '/media/database/data2/Expression/BP4D/AUCoding/AU_OCC'
    task2emo = {
            'T1': 'happy',
            'T2': 'sad',
            'T3': 'surprise',
            'T4': 'embarrassment',
            'T5': 'fear',
            'T6': 'physical pain',
            'T7': 'anger',
            'T8': 'disgust',
        }
    tasks = list(task2emo.keys())
    expressions = list(task2emo.values())
    emo2task = dict(zip(expressions, tasks))
    taskindex = dict(zip(tasks, range(len(tasks)))) #通过名称找到index
    exindex = dict(zip(expressions, range(len(expressions))))
    stat = {}
    for i in range(len(task2emo)):
        AU_cnt = dict(zip(list(range(0, len(AU)+1)), [0]*len(AU)))
        stat[tasks[i]] = AU_cnt

    file_list = os.listdir(path_info)
    for file in file_list:
        # 当前数据的表情标签由文件标题中的任务决定
        task = file.split('.')[0].split('_')[-1]

        cur_info_path = os.path.join(path_info, file)
        file_info = np.array(pd.read_csv(cur_info_path)) # 文件中AU的发生情况从第二列起，第一列为frame编码
        for i in range(file_info.shape[0]):
            cur_item = file_info[i, AU]
            cur_item = np.where(cur_item != 9, cur_item, 0)
            mask_info = np.array(np.nonzero(cur_item)).squeeze(0).tolist()
            if len(mask_info) > 1:
                stat[task][0] += 1
                for j1 in mask_info:
                    stat[task][j1+1] += 1

    EMO2AU = np.zeros((len(tasks), len(AU)))
    for i in range(EMO2AU.shape[0]):
        for j in range(1, EMO2AU.shape[1]):
            EMO2AU[i][j] = stat[tasks[i]][j] / stat[tasks[i]][0]
    EMO2AU_re = np.zeros((len(EMO), len(AU)))
    for emo_i, emo_name in enumerate(EMO):
        task = emo2task[emo_name]
        task_i = taskindex[task]
        EMO2AU_re[emo_i, :] = EMO2AU[task_i, :]
    return EMO2AU_re, EMO, AU


def test(AU, EMO, AU_cpt, prob_AU, EMO2AU):
    global device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = utils.getDatasetInfo('BP4D')
    ACC = []
    for k, emo in enumerate(EMO):
        acc_record = []
        for idx, (cur_item, emo_label, index) in enumerate(test_loader[k], 1):
            weight = []
            occ_au = []
            prob_all_au = np.zeros((len(AU),))
            evidence = {}
            
            for i, au in enumerate(AU):
                if cur_item[0, au] == 1:
                    prob_all_au[i] = 1
                    evidence['AU'+str(au)] = 1
                    occ_au.append(i)
                    evidence['AU'+str(au)] = 1
                weight.append(EMO2AU[:, i])

            if len(occ_au) >= 2:
                pos = np.where(prob_all_au > 0.6)[0]
                weight = np.array(weight)
                
                for i in range(prob_all_au.shape[0]-2):
                    if i in pos:
                        prob_all_au[i] = prob_all_au[i] / prob_AU[i]
                    else:
                        prob_all_au[i] = 1 / (1-prob_AU[i])
                        weight[i, :] = 1 - weight[i, :]

                weight = np.where(weight > 0, weight, 1e-5)
                update = UpdateGraph(in_channels=1, out_channels=len(EMO), W=weight, prob_all_au=prob_all_au).to(device)

                AU_evidence = torch.ones((1, 1)).to(device)
                cur_prob = update(AU_evidence)
                cur_pred = torch.argmax(cur_prob)

                acc_record.append(torch.eq(cur_pred, emo_label.to(device)).type(torch.FloatTensor).item())
                ACC.append(torch.eq(cur_pred, emo_label.to(device)).type(torch.FloatTensor).item())
        print('Acc of %s is %.5f' %(emo, np.array(acc_record).mean()))
    print('The dataset Acc of %s is %.5f' %(emo, np.array(ACC).mean()))


if __name__ == '__main__':
    AU_cpt, prob_AU = interAU()
    EMO2AU, EMO, AU = dataset_EMO2AU()
    test(AU, EMO, AU_cpt, prob_AU, EMO2AU)
    a = 1

    '''
    Train set size: 96375
    Test set size: 10711
    Acc of happy is 1.00000
    Acc of sad is 0.00000
    Acc of fear is 0.00000
    Acc of anger is 0.00000
    Acc of surprise is 0.00000
    Acc of disgust is 0.00000
    The dataset Acc of disgust is 0.21471
    '''

