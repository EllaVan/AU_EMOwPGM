'''
包括最简单的tesnt_rules_throw(给定阈值, 把unseen抛出去); test_rules_dis(用欧式距离分类)
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from conf import ensure_dir

from tensorboardX import SummaryWriter

# from models.AU_EMO_BP import UpdateGraph_v2 as UpdateGraph
from models.RadiationAUs import RadiateAUs_v2 as RadiateAUs
from utils import *

import sys
from torch.autograd import Variable

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np

# from models import AU_EMO_bayes
from functools import reduce
import logging

def test_rules_throw(conf, update, device, input_info, input_rules, AU_p_d, summary_writer, confu_m=None):
    priori_AU, dataset_AU = AU_p_d
    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    
    criterion = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    # num_EMO = EMO2AU_cpt.shape[0]
    num_EMO = max(labelsEMO)+1
    if confu_m is None:
        confu_m = torch.zeros((num_EMO+1, num_EMO+1))
        # confu_m = torch.zeros((num_EMO, num_EMO))

    device = conf.device
    loc1 = conf.loc1
    loc2 = conf.loc2

    # update = UpdateGraph(conf, EMO2AU_cpt, prob_AU, loc1, loc2).to(device)
    update.eval()
    init_label_record = []
    init_pred_record = []
    factor = (1/6)**len(AU)
    dataset_AU = priori_AU

    with torch.no_grad():
        for idx in range(labelsAU.shape[0]):
            torch.cuda.empty_cache()
            cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
            emo_label = labelsEMO[idx].reshape(1,).to(device)

            occ_au = []
            for priori_au_i, priori_au in enumerate(priori_AU):
                if priori_au in dataset_AU:
                    pos_priori_in_data = dataset_AU.index(priori_au)
                    if cur_item[0, pos_priori_in_data] == 1:
                        occ_au.append(priori_au_i)

            if cur_item.sum() != 0:
                if emo_label <= 5:
                    if_static_op = True
                else:
                    if_static_op = False
                prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU_cpt, thresh=0.6, if_static_op=False)
                cur_prob, _ = update(prob_all_au)
                cur_pred = torch.argmax(cur_prob)

                if update.out2[0][cur_pred] >= 0.9:
                    # confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
                    # err = criterion(cur_prob, emo_label)
                    # acc = torch.eq(cur_pred, emo_label).sum().item()
                    # err_record.append(err.item())
                    # acc_record.append(acc)
                    # summary_writer.add_scalar('val_err', np.array(err_record).mean(), idx)
                    # summary_writer.add_scalar('val_acc', np.array(acc_record).mean(), idx)
                    # torch.cuda.empty_cache()
                    # init_label_record.append(update.out2[0][emo_label])
                    # init_pred_record.append(update.out2[0][cur_pred])
                    # del prob_all_au, cur_prob, cur_pred#, err, acc
                    pass
                else:
                    confu_m = confusion_matrix(preds=cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=[-1], conf_matrix=confu_m)

    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    return output_records

def test_rules_dis(conf, update, device, input_info, input_rules, AU_p_d, summary_writer, confu_m=None):
    priori_AU, dataset_AU = AU_p_d
    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    num_EMO = EMO2AU_cpt.shape[0]
    
    criterion = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    if confu_m is None:
        confu_m = torch.zeros((num_EMO, num_EMO))

    device = conf.device
    loc1 = conf.loc1
    loc2 = conf.loc2

    # update = UpdateGraph(conf, EMO2AU_cpt, prob_AU, loc1, loc2).to(device)
    update.eval()

    with torch.no_grad():
        for idx in range(labelsAU.shape[0]):
            torch.cuda.empty_cache()
            cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
            emo_label = labelsEMO[idx].reshape(1,).to(device)

            occ_au = []
            for priori_au_i, priori_au in enumerate(priori_AU):
                if priori_au in dataset_AU:
                    pos_priori_in_data = dataset_AU.index(priori_au)
                    if cur_item[0, pos_priori_in_data] == 1:
                        occ_au.append(priori_au_i)

            if len(occ_au) > 1: # len(occ_au) != 0
                prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU_cpt, thresh=0.6)

                prob_all_au_r = prob_all_au.reshape(1, -1).repeat(num_EMO, 0)
                dis_matrix = EMO2AU_cpt - prob_all_au_r
                dis = np.abs(dis_matrix).sum(axis=1).reshape(num_EMO, )
                sum_dis = sum(dis)
                cur_prob = np.zeros((num_EMO, ))
                for i in range(cur_prob.shape[0]):
                    cur_prob[i] = dis[i] / sum_dis
                cur_pred = np.argmax(cur_prob)
                confu_m = confusion_matrix(cur_pred.reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)

                cur_prob = torch.from_numpy(cur_prob).to(device)
                cur_pred = torch.tensor(cur_pred).to(device)
                # err = criterion(cur_prob, emo_label)
                acc = torch.eq(cur_pred, emo_label).sum().item()
                # err_record.append(err.item())
                acc_record.append(acc)
                # summary_writer.add_scalar('val_err', np.array(err_record).mean(), idx)
                summary_writer.add_scalar('val_acc', np.array(acc_record).mean(), idx)

                del prob_all_au, cur_prob, cur_pred, acc#, err
    if len(acc_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    return output_records