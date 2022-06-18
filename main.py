import argparse
import datetime
import pytz
import os
import shutil
import pickle as pkl

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
    train_loader, test_loader = utils.getDatasetInfo(args.dataset)
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())

    summary_writer = SummaryWriter(args.save_path)
    
    # 建立索引和EMO/AU名称的双向映射
    index2EMO_dict = dict(zip(range(len(EMO)), EMO))
    EMO2index_dict = dict(zip(EMO, range(len(EMO)))) #通过名称找到index
    AU2index_dict = dict(zip(list(map(int, AU)), range(len(AU))))
    index2AU_dict = dict(zip(range(len(AU)), list(map(int, AU))))

    EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, 1e-4)

    save_pkl = {}
    save_pkl['ori_EMO2AU'] = EMO2AU_cpt

    # 交叉熵损失函数
    CE = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    AU_occ = dict(zip(AU, [0]*len(AU)))
    a = 0
    AU = AU[:-2]
    
    for idx, (cur_item, emo_label, index) in enumerate(train_loader, 1):
        weight = []
        occ_au = []
        abs_au = []
        
        for i, au in enumerate(AU):
            if cur_item[0, au] == 1:
                occ_au.append(i)
                weight.append(EMO2AU_cpt[:, i])
                AU_occ[au] += 1
            elif cur_item[0, au] == 0:
                abs_au.append(i)
                weight.append(np.array([1-value for value in list(EMO2AU_cpt[:, i])]))

        if len(occ_au) != 0:
            
            if emo_label == 0:
                weight.append(EMO2AU_cpt[:, -2])
                weight.append(np.array([1-value for value in list(EMO2AU_cpt[:, -1])]))
            elif emo_label == 2 or emo_label == 4:
                weight.append(EMO2AU_cpt[:, -2])
                weight.append(EMO2AU_cpt[:, -1])
            elif emo_label == 5:
                weight.append(np.array([1-value for value in list(EMO2AU_cpt[:, -2])]))
                weight.append(EMO2AU_cpt[:, -1])
            else:
                weight.append(np.array([1-value for value in list(EMO2AU_cpt[:, -2])]))
                weight.append(np.array([1-value for value in list(EMO2AU_cpt[:, -1])]))

            weight = np.array(weight)
            weight = np.where(weight > 0, weight, 1e-4)
            update = UpdateGraph(in_channels=1, out_channels=len(EMO), W=weight).to(device)
            optim_graph = optim.SGD(update.parameters(), lr=args.lr)
            
            AU_evidence = torch.ones((1, 1)).to(device)
            cur_prob = update(AU_evidence)
            cur_pred = torch.argmax(cur_prob)

            err = CE(cur_prob, emo_label.to(device))
            err_record.append(utils.loss_to_float(err))
            acc_record.append(torch.eq(cur_pred, emo_label.to(device)).type(torch.FloatTensor).item())

            summary_writer.add_scalar('err', np.array(err_record).mean(), idx)
            summary_writer.add_scalar('err_single', utils.loss_to_float(err), idx)
            summary_writer.add_scalar('acc', np.array(acc_record).mean(), idx)

            optim_graph.zero_grad()
            err.backward()
            optim_graph.step()
                        
            new_EMO2AU_cpt = EMO2AU_cpt
            update_info = update.fc.weight.grad.cpu().numpy()
            summary_writer.add_scalar('update_info', update_info[emo_label], idx)
            for i, j in enumerate(AU):
                factor = 1
                if i in occ_au:
                    new_EMO2AU_cpt[emo_label, i] += args.lr*np.abs(update_info[emo_label]*factor)
                else:
                    new_EMO2AU_cpt[emo_label, i] -= args.lr*np.abs(update_info[emo_label]*factor)
            EMO2AU_cpt_tmp = np.array(EMO2AU_cpt)

            EMO2AU_cpt_tmp = np.where(EMO2AU_cpt_tmp > 0, EMO2AU_cpt_tmp, 1e-4)
            EMO2AU_cpt = np.where(EMO2AU_cpt_tmp <= 1, EMO2AU_cpt_tmp, 1)
            
            if idx == 10000:
                args.lr /= 10.0

    save_pkl['new_EMO2AU'] = EMO2AU_cpt
    save_pkl['AU_occ'] = AU_occ
    save_pkl['train_size'] = len(train_loader)
    with open(os.path.join(args.save_path, 'results.pkl'), 'wb') as fo:
        pkl.dump(save_pkl, fo)
    fo.close()
    print(a)
    

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
