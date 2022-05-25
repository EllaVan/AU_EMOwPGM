import argparse
import datetime
import pytz
import os
import shutil

import csv
import pandas as pd
import numpy as np
import torch

from materials.process_priori import cal_interAUPriori
from models.AU_EMO_bayes import initGraph
from InferPGM.inference import VariableElimination


def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--dataset', type=str, default='BP4D')
    parser.add_argument('--save_path', type=str, default='save')

    # --------------settings for feature extraction-------------
    parser.add_argument('--manualSeed', type=int, default=None)
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=0)
    parser.add_argument('--save_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--lr', type=float, default=0.001)
    
    return parser.parse_args()


def Update(init_AU_EMO_cpt, init_prob_AU, init_EMO_img_num, init_AU_cpt):
    pass


def run(args):
    init_AU_EMO_cpt, init_prob_AU, init_EMO_img_num, init_AU_cpt, EMO, AU = cal_interAUPriori()
    c = list(map(int, AU))
    AU = c
    
    # 建立索引和EMO/AU名称的双向映射
    index2EMO_dict = dict(zip(range(len(EMO)), EMO))
    EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index
    AU2index_dict = dict(zip(list(map(int, AU)), range(len(AU))))
    index2AU_dict = dict(zip(range(len(AU)), list(map(int, AU))))

    if args.dataset == 'BP4D':
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
        AU_mask = np.array([0]*26)
        for i in AU:
            AU_mask[i-1] = 1
    
    AU_EMO_model = initGraph()
    model_infer = VariableElimination(AU_EMO_model)
    
    file_list = os.listdir(path_info)
    for file in file_list:
        cur_info_path = os.path.join(path_info, file)

        outfile = os.path.join(args.save_path, file.split('.')[0] + 'pred.csv')
        header = ['idx'] + EMO + ['Result']
        with open(outfile, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
        emo_label = EMO2index_dict[EMO_code_dict[file.split('.')[0].split('_')[-1]]]
        AU_evidence = {}
        cur_row = pd.read_csv(cur_info_path)
        cur_info = np.array(cur_row.iloc[:, 1:])

        for i in range(cur_info.shape[0]):
            cur_item = np.array(cur_info[i])
            for j in AU:
                AU_evidence['AU'+str(j)] = cur_item[j-1]
                if cur_item[j-1] == 9:
                    AU_evidence['AU'+str(j)] = 0
            if emo_label == 0 :
                AU_evidence['AU26'] = 1
            elif emo_label == 2 or emo_label == 4:
                AU_evidence['AU25'] = 1
                AU_evidence['AU26'] = 1
            elif emo_label == 5:
                AU_evidence['AU25'] = 1
            
            q = model_infer.query(variables=['EMO'], evidence=AU_evidence)

            cur_idx = [cur_row.iloc[i, 0]]
            cur_prob = list(q.values)
            cur_out = [index2EMO_dict[np.argmax(q.values)]]
            cur_pred = cur_idx + cur_prob + cur_out
            with open(outfile, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(cur_pred)


if __name__ == '__main__':
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)

    args = parse_args()

    cur_day = str(cur_time).split(' ')
    cur_day = cur_day[0]
    args.save_path = os.path.join(args.save_path, cur_day)
    if os.path.exists(args.save_path) is True:
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)
    else:
        os.makedirs(args.save_path)

    print('%s based Update' % (args.dataset))
    run(args)
