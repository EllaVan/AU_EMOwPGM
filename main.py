import argparse
import datetime
import pytz
import os
import shutil

import csv
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from materials.process_priori import cal_interAUPriori
from models.AU_EMO_BP import UpdateGraph
from models.AU_EMO_bayes import initGraph, AU_EMO_bayesGraph
from InferPGM.inference import VariableElimination

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='BP4D')
    parser.add_argument('--save_path', type=str, default='save')

    # --------------settings for feature extraction-------------
    parser.add_argument('--manualSeed', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=0)
    parser.add_argument('--save_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--lr', type=float, default=0.001)
    
    return parser.parse_args()


def run_myIdea(args):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    
    # 建立索引和EMO/AU名称的双向映射
    index2EMO_dict = dict(zip(range(len(EMO)), EMO))
    EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index
    AU2index_dict = dict(zip(list(map(int, AU)), range(len(AU))))
    index2AU_dict = dict(zip(range(len(AU)), list(map(int, AU))))

    path_info, EMO_code_dict = utils.getDatasetInfo(args.dataset)

    AU_EMO_jpt = EMO2AU_cpt*1.0/len(EMO)
    AU2EMO_cpt = []
    for i in range(len(AU)):
        AU2EMO_cpt.append(list(AU_EMO_jpt[:, i]/prob_AU[i]))
    AU2EMO_cpt = np.array(AU2EMO_cpt)

    # 交叉熵损失函数
    CE = nn.CrossEntropyLoss()
    
    # 按训练数据集路径开始训练
    file_list = os.listdir(path_info)
    end_flag = 0
    for file in file_list:
        end_flag += 1 # 对训练文件计数
        
        # 当前数据的表情标签由文件标题中的任务决定
        task = file.split('.')[0].split('_')[-1]
        if task != 'T4' and task != 'T6': # 暂时不考虑基本表情以外的任务
            emo_name = EMO_code_dict[task] 
            emo_label = EMO2index_dict[emo_name]

            # 建立输出文件
            log_dir = os.path.join(args.save_path, file.split('.')[0])
            err_writer = SummaryWriter(log_dir)
            outfile = os.path.join(log_dir, 'pred_Idea.csv')
            header = ['idx'] + EMO + ['Result='+emo_name]
            with open(outfile, 'w', encoding='UTF8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(header)     
            
            cur_info_path = os.path.join(path_info, file)
            file_info = np.array(pd.read_csv(cur_info_path)) # 文件中AU的发生情况从第二列起，第一列为frame编码
            err_record = []
            for frame in range(file_info.shape[0]): # 遍历每一帧
                cur_item = np.array(file_info[frame, :])
                x = []
                weight = []
                au_item = []
                
                for i, au in enumerate(AU):
                    if cur_item[au] == 1:
                        x.append(1.0)
                        au_item.append(i)
                        weight.append(AU2EMO_cpt[i, :])
                if emo_label == 0:
                    x.append(1.0)
                    weight.append(AU2EMO_cpt[-2, :])
                elif emo_label == 2 or emo_label == 4:
                    x.append(1.0)
                    x.append(1.0)
                    weight.append(AU2EMO_cpt[-1, :])
                    weight.append(AU2EMO_cpt[-2, :])
                elif emo_label == 5:
                    x.append(1.0)
                    weight.append(AU2EMO_cpt[-1, :])

                if len(x) != 0:
                    weight = np.array(weight)
                    update = UpdateGraph(weight.shape[0], len(EMO), weight).to(device)
                    optim_graph = optim.SGD(update.parameters(), lr=0.0001)
                    
                    AU_evidence = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                    cur_prob = update(AU_evidence)

                    err = CE(cur_prob, torch.tensor(emo_label, dtype=torch.long).unsqueeze(0).to(device))
                    err_record.append(utils.loss_to_float(err))
                    
                    cur_idx = [file_info[frame, 0]] # 当前帧的编码
                    cur_pred = [index2EMO_dict[int(torch.argmax(cur_prob))]] # 由PGM + AU序列推理出的可能的EMO状态号
                    cur_out = cur_idx + cur_prob.tolist() + cur_pred
                    with open(outfile, 'a', encoding='UTF8', newline='') as f: # 写入PGM推理的结果
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(cur_out)

                    err_writer.add_scalar('err', np.array(err.detach().cpu()), frame)
                    optim_graph.zero_grad()
                    err.backward()
                    optim_graph.step()

                    new_AU2EMO_cpt = AU2EMO_cpt
                    update_info = update.fc.weight.T.detach().cpu().numpy()
                    for i, j in enumerate(au_item):
                        new_AU2EMO_cpt[j, emo_label] = update_info[i, emo_label]
                    AU2EMO_cpt_tmp = np.array(new_AU2EMO_cpt)
                    AU2EMO_cpt = np.where(AU2EMO_cpt_tmp >= 0, AU2EMO_cpt_tmp, 0)
                    AU2EMO_cpt = np.where(AU2EMO_cpt_tmp <= 1, AU2EMO_cpt_tmp, 1)

    break_flag = True


if __name__ == '__main__':
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)

    args = parse_args()

    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print('using gpu:', args.gpu)

    cur_day = str(cur_time).split(' ')
    cur_day = cur_day[0]
    args.save_path = os.path.join(args.save_path, cur_day)
    if os.path.exists(args.save_path) is True:
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)
    else:
        os.makedirs(args.save_path)

    print('%s based Update' % (args.dataset))
    run_myIdea(args)
