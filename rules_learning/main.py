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

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.utils import get_example_model

from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs, interAUs
import rules_learning.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='BP4D')
    parser.add_argument('--save_path', type=str, default='save')

    # --------------------settings for training-------------------
    parser.add_argument('--manualSeed', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=0)
    parser.add_argument('--save_epoch', type=int, default=0)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_idx', type=int, default=20000)
    parser.add_argument('--AUthresh', type=float, default=0.6)
    parser.add_argument('--zeroPad', type=float, default=1e-4)
    
    return parser.parse_args()


def run_myIdea(args):
    train_loader, test_loader = utils.getDatasetInfo(args.dataset)
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
    
    summary_writer = SummaryWriter(args.save_path)

    nodes = ['AU'+str(au) for au in AU]
    ori_size = np.sum(np.array(EMO_img_num))
    num_all_img = ori_size
    AU_cpt = AU_cpt - np.eye(len(AU))
    AU_ij_cnt = AU_cpt * num_all_img
    AU_cnt = prob_AU * num_all_img

    save_pkl = {}
    save_pkl['ori_EMO2AU'] = EMO2AU_cpt
    save_pkl['ori_AU_cpt'] = AU_cpt
    save_pkl['ori_prob_AU'] = prob_AU
    
    # 交叉熵损失函数与输出记录
    CE = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    lr_flag = 0
    if_bonus = False
    AU_evidence = torch.ones((1, 1)).to(device)
    
    for idx, (cur_item, emo_label, index) in enumerate(train_loader, 1):
        
        torch.cuda.empty_cache()
        variable = nodes.copy()
        weight = []
        occ_au = []
        evidence = {}
        prob_all_au = np.zeros((len(AU),))
        
        if len(cur_item.shape) > 2:
            cur_item = cur_item.squeeze(0)
        cur_item = np.where(cur_item != 9, cur_item, 0)
        for i, au in enumerate(AU):
            if cur_item[0, au] == 1:
                occ_au.append(i)
                prob_all_au[i] = 1
                AU_cnt[i] += 1
                evidence['AU'+str(au)] = 1
                variable.remove('AU'+str(au))
            weight.append(EMO2AU_cpt[:, i])

        if len(occ_au) != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=args.AUthresh)
            
            pos = np.where(prob_all_au > args.AUthresh)[0] # pos = np.where(prob_all_au == 1)[0]
            weight = np.array(weight)
            
            for i in range(prob_all_au.shape[0]-2):
                if i in pos:
                    prob_all_au[i] = prob_all_au[i] / prob_AU[i]
                else:
                    prob_all_au[i] = 1 / (1-prob_AU[i])
                    weight[i, :] = 1 - weight[i, :]

            prob_all_au[-2] = 1 / prob_AU[-2]
            prob_all_au[-1] = 1 / prob_AU[-1]
            if emo_label == 0:
                weight[-1, :] = 1 - weight[-1, :]
                prob_all_au[-1] = 1 / (1-prob_AU[-1])
            elif emo_label == 2 or emo_label == 4:
                pass
            else:
                weight[-2, :] = 1 - weight[-2, :]
                weight[-1, :] = 1 - weight[-1, :]
                prob_all_au[-2] = 1 / (1-prob_AU[-2])
                prob_all_au[-1] = 1 / (1-prob_AU[-1])
            init = np.ones((1, len(EMO)))
            for i in range(weight.shape[1]):
                for j in range(1, 3):
                    init[:, i] *= weight[-j][i]*prob_all_au[-j]
            
            weight = np.where(weight > 0, weight, args.zeroPad)
            update = UpdateGraph(device, in_channels=1, out_channels=len(EMO), W=weight[:-2, :], 
                                prob_all_au=prob_all_au[:-2], init=init).to(device)
            optim_graph = optim.SGD(update.parameters(), lr=args.lr)
            
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
            
            new_EMO2AU_cpt = EMO2AU_cpt # torch.Tensor(EMO2AU_cpt).to(device)
            # weight = torch.Tensor(weight).to(device)
            update_info1 = update.fc.weight.grad.cpu().numpy().squeeze() # update.fc.weight.grad.squeeze()
            update_info2 = update.d1.detach().cpu().numpy().squeeze() # update.d1.detach().squeeze()
            for emo_i, emo_name in enumerate(EMO):
                if emo_i == emo_label:
                    for i, j in enumerate(AU[:-2]):
                        factor = update_info2[emo_i] / weight[i, emo_i]
                        weight[i, emo_i] -= args.lr*update_info1[emo_i]*factor
                        if i in pos:
                            new_EMO2AU_cpt[emo_i, i] = weight[i, emo_i] # new_EMO2AU_cpt[emo_label, i] += args.lr*np.abs(update_info1[emo_label]*factor)
                        else:
                            new_EMO2AU_cpt[emo_i, i] = 1-weight[i, emo_i] # new_EMO2AU_cpt[emo_label, i] -= args.lr*np.abs(update_info1[emo_label]*factor)
                else:
                    for i, j in enumerate(AU[:-2]):
                        factor = update_info2[emo_i] / weight[i, emo_i]
                        weight[i, emo_i] -= args.lr*update_info1[emo_i]*factor
                        if i in pos:
                            new_EMO2AU_cpt[emo_i, i] = weight[i, emo_i]
                        else:
                            new_EMO2AU_cpt[emo_i, i] = 1-weight[i, emo_i]

            EMO2AU_cpt = np.array(new_EMO2AU_cpt) # new_EMO2AU_cpt.cpu().numpy()
            EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, args.zeroPad)
            EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)

            for i, au_i in enumerate(occ_au):
                for j, au_j in enumerate(occ_au):
                    if i != j:
                        AU_ij_cnt[au_i][au_j] += 1
                        AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
            for i, j in enumerate(AU[:-2]):
                prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            prob_AU = np.where(prob_AU > 0, prob_AU, args.zeroPad)
            prob_AU = np.where(prob_AU <= 1, prob_AU, 1)

            if idx >= args.lr_decay_idx and lr_flag == 0:
                lr_flag = 1
                args.lr /= 10.0
    
    save_pkl['new_EMO2AU'] = EMO2AU_cpt
    save_pkl['new_AU_cpt'] = AU_cpt
    save_pkl['train_size'] = num_all_img - ori_size
    save_pkl['new_prob_AU'] = prob_AU
    save_pkl['AU_cnt'] = AU_cnt
    save_pkl['AU_ij_cnt'] = AU_ij_cnt
    with open(os.path.join(args.save_path, 'results.pkl'), 'wb') as fo:
        pkl.dump(save_pkl, fo)
    fo.close()
    
    print('Training Done')
    

    end_flag = True 


if __name__ == '__main__': 
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)

    args = parse_args()

    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print('using gpu:', args.gpu)

    cur_day = str(cur_time).split(' ')
    cur_day = cur_day[0]
    args.save_path = os.path.join(args.save_path, args.dataset, cur_day)
    if os.path.exists(args.save_path) is True:
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)
    else: 
        os.makedirs(args.save_path)

    print('%s based Update' % (args.dataset))
    torch.cuda.empty_cache()
    run_myIdea(args)
    