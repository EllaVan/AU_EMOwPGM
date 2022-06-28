import argparse
import datetime
from errno import EMULTIHOP
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

from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs
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
    parser.add_argument('--AUthresh', type=float, default=0.6)
    parser.add_argument('--zeroPad', type=float, default=1e-4)
    
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

    num_all_img = np.sum(np.array(EMO_img_num))
    # AU = AU[:-2]
    AU_cpt = AU_cpt - np.eye(len(AU))
    AU_ij_cnt = AU_cpt * num_all_img
    AU_cnt = prob_AU * num_all_img
    # EMO2AU_cpt = EMO2AU_cpt[:, :-2]
    # prob_AU = prob_AU[:-2]

    save_pkl = {}
    save_pkl['ori_EMO2AU'] = EMO2AU_cpt
    save_pkl['ori_AU_cpt'] = AU_cpt
    save_pkl['ori_prob_AU'] = prob_AU

    # 交叉熵损失函数
    CE = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    
    for idx, (cur_item, emo_label, index) in enumerate(train_loader, 1):
        weight = []
        occ_au = []
        
        for i, au in enumerate(AU):
            if cur_item[0, au] == 1:
                occ_au.append(i)
                AU_cnt[i] += 1
            weight.append(EMO2AU_cpt[:, i])

        if len(occ_au) >= 4:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=args.AUthresh)
            pos = np.where(prob_all_au == 1)[0] # pos = np.where(prob_all_au > args.AUthresh)[0]
            weight = np.array(weight)
            
            prob_all_au[-2] = 1 / prob_AU[-2]
            prob_all_au[-1] = 1 / prob_AU[-1]
            if emo_label == 0:
                weight[-1, :] = 1 - weight[-1, :]
            elif emo_label == 2 or emo_label == 4:
                pass
            elif emo_label == 5:
                weight[-2, :] = 1 - weight[-2, :]
            else:
                weight[-2, :] = 1 - weight[-2, :]
                weight[-1, :] = 1 - weight[-1, :]
            
            for i in range(prob_all_au.shape[0]-2):
                if i in pos:
                    prob_all_au[i] = prob_all_au[i] / prob_AU[i]
                else:
                    prob_all_au[i] = 1 / prob_AU[i]
                    weight[i, :] = 1 - weight[i, :]
            
            prob_all_au_cp = prob_all_au.copy()
            weight = np.where(weight > 0, weight, args.zeroPad)
            update = UpdateGraph(in_channels=1, out_channels=len(EMO), W=weight, prob_all_au=prob_all_au).to(device)
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
            update_info1 = update.fc.weight.grad.cpu().numpy()
            update_info2 = update.d1.cpu().numpy().squeeze()
            for i, j in enumerate(AU[:-2]):
                # factor = prob_all_au_cp[i]
                factor = update_info2[emo_label] / weight[i, emo_label]
                if i in pos:
                    new_EMO2AU_cpt[emo_label, i] += args.lr*np.abs(update_info1[emo_label]*factor)
                else:
                    new_EMO2AU_cpt[emo_label, i] -= args.lr*np.abs(update_info1[emo_label]*factor)

            if emo_label == 0:
                factor = update_info2[emo_label] / weight[-2, emo_label]
                new_EMO2AU_cpt[emo_label, -2] += args.lr*np.abs(update_info1[emo_label]*factor)
                factor = update_info2[emo_label] / weight[-1, emo_label]
                new_EMO2AU_cpt[emo_label, -1] -= args.lr*np.abs(update_info1[emo_label]*factor)
            elif emo_label == 2 or emo_label == 4:
                factor = update_info2[emo_label] / weight[-2, emo_label]
                new_EMO2AU_cpt[emo_label, -2] += args.lr*np.abs(update_info1[emo_label]*factor)
                factor = update_info2[emo_label] / weight[-1, emo_label]
                new_EMO2AU_cpt[emo_label, -1] += args.lr*np.abs(update_info1[emo_label]*factor)
            elif emo_label == 5:
                factor = update_info2[emo_label] / weight[-2, emo_label]
                new_EMO2AU_cpt[emo_label, -2] -= args.lr*np.abs(update_info1[emo_label]*factor)
                factor = update_info2[emo_label] / weight[-1, emo_label]
                new_EMO2AU_cpt[emo_label, -1] += args.lr*np.abs(update_info1[emo_label]*factor)
            else:
                factor = update_info2[emo_label] / weight[-2, emo_label]
                new_EMO2AU_cpt[emo_label, -2] -= args.lr*np.abs(update_info1[emo_label]*factor)
                factor = update_info2[emo_label] / weight[-1, emo_label]
                new_EMO2AU_cpt[emo_label, -1] -= args.lr*np.abs(update_info1[emo_label]*factor)
            
            EMO2AU_cpt = np.array(new_EMO2AU_cpt)
            EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, 1e-4)
            EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)

            for i, au_i in enumerate(occ_au):
                for j, au_j in enumerate(occ_au):
                    if i != j:
                        AU_ij_cnt[au_i][au_j] += 1
                        AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
            for i, j in enumerate(AU):
                prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            prob_AU = np.where(prob_AU > 0, prob_AU, args.zeroPad)
            prob_AU = np.where(prob_AU <= 1, prob_AU, 1)
            if idx == 20000:
                args.lr /= 10.0

    save_pkl['new_EMO2AU'] = EMO2AU_cpt
    save_pkl['new_AU_cpt'] = AU_cpt
    save_pkl['train_size'] = len(train_loader)
    save_pkl['new_prob_AU'] = prob_AU
    save_pkl['AU_cnt'] = AU_cnt
    save_pkl['AU_ij_cnt'] = AU_ij_cnt
    with open(os.path.join(args.save_path, 'results.pkl'), 'wb') as fo:
        pkl.dump(save_pkl, fo)
    fo.close()

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
    args.save_path = os.path.join(args.save_path, cur_day)
    if os.path.exists(args.save_path) is True:
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)
    else:
        os.makedirs(args.save_path)

    print('%s based Update' % (args.dataset))
    run_myIdea(args)
    