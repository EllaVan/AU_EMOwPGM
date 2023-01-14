import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from conf import ensure_dir
from models.AU_EMO_BP import UpdateGraph_continuous as UpdateGraph
from models.RadiationAUs import RadiateAUs_v2 as RadiateAUs
from models.focal_loss import MultiClassFocalLossWithAlpha
from utils import *
import logging

def crop_EMO2AU(conf, priori_update, *args):
    EMO2AU_cpt = priori_update.EMO2AU_cpt.data.detach().cpu().numpy()
    prob_AU = priori_update.prob_AU.data.detach().cpu().numpy()
    EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
    EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)
    priori_update.EMO2AU_cpt.data.copy_(torch.from_numpy(EMO2AU_cpt))
    loc1 = priori_update.loc1
    loc2 = priori_update.loc2
    for i, j in enumerate(loc1):
        prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (EMO2AU_cpt.shape[0])
    priori_update.prob_AU.data.copy_(torch.from_numpy(prob_AU))
    if len(args) != 0:
        occ_au, AU_ij_cnt, AU_cpt, AU_cnt = args
        for i, au_i in enumerate(occ_au):
            for j, au_j in enumerate(occ_au):
                if i != j:
                    AU_ij_cnt[au_i][au_j] = AU_ij_cnt[au_i][au_j]+1
                    AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
        return priori_update, AU_ij_cnt, AU_cpt, AU_cnt
    else:
        return priori_update

def final_return(priori_update, EMO, AU, loc1, loc2):
    EMO2AU_cpt = np.zeros((len(EMO), len(AU)))
    EMO2AU_cpt1 = priori_update.EMO2AU_cpt.data.detach().cpu().numpy()
    EMO2AU_cpt2 = priori_update.static_EMO2AU_cpt.data.detach().cpu().numpy()
    EMO2AU_cpt[:, loc1] = EMO2AU_cpt1
    EMO2AU_cpt[:, loc2] = EMO2AU_cpt2
    prob_AU = np.zeros((len(AU),))
    prob_AU1 = priori_update.prob_AU.data.detach().cpu().numpy()
    prob_AU2 = priori_update.static_prob_AU.data.detach().cpu().numpy()
    prob_AU[loc1] = prob_AU1
    prob_AU[loc2] = prob_AU2
    return priori_update, EMO2AU_cpt, prob_AU

def learn_rules(conf, input_info, input_rules, seen_trained_rules, AU_p_d, summary_writer, *args):
    device = conf.device
    loc1 = conf.loc1
    loc2 = conf.loc2
    labelsAU, labelsEMO = input_info
    priori_AU, dataset_AU = AU_p_d

    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    train_size = labelsAU.shape[0]
    EMO2AU = EMO2AU_cpt

    seen_trained_EMO2AU_cpt, seen_trained_AU_cpt, seen_trained_prob_AU, seen_trained_ori_size, seen_trained_num_all_img, seen_trained_AU_ij_cnt, seen_trained_AU_cnt, seen_EMO, AU = seen_trained_rules
    num_seen = seen_trained_EMO2AU_cpt.shape[0]
    num_unseen = EMO2AU_cpt.shape[0] - num_seen
    target_seen = seen_trained_EMO2AU_cpt[:, loc1]
    target_seen = torch.from_numpy(target_seen).to(device)
    # seen_trained_model = UpdateGraph(conf, seen_trained_rules, temp=2).to(device)
    # seen_trained_model.eval()
    cri_KL = nn.KLDivLoss(reduction='batchmean')
    KL_record = []

    init_lr = conf.lr_relation
    if init_lr == -1:
        if train_size > num_all_img:
            init_lr = num_all_img / (num_all_img + train_size)
        else:
            init_lr = train_size / (num_all_img + train_size)
        
    if args:
        change_weight2 = 1
        for changing_item in args:
            change_weight2 = change_weight2 * changing_item
        init_lr = init_lr * change_weight2

    loss_weight = []
    cl_num = []
    pre_weight = train_size/len(EMO)
    for emoi, emon in enumerate(EMO):
        cl_weight = torch.where(labelsEMO==emoi)[0].shape[0] # 每种标签有多少个样本
        cl_num.append(cl_weight)
        loss_weight.append(cl_weight/pre_weight) # N_mean / N = init_lr / cur_lr, N_mean表示总样本数量下保持EMO均匀分布的平均样本数量
    t1 = sum(cl_num)

    # for each_weighti in range(len(cl_num)):
    #     # Focal_Loss中的alpha参数越大，标签为这一类的loss值越大，为了降低样本数量多的类别的loss影响，需要为样本量多的类别赋小权重
    #     cl_num[each_weighti] = (t1-cl_num[each_weighti])/t1

    t2 = sum(cl_num[:num_seen])
    for each_weighti, each_weight in enumerate(cl_num):
        # (扩展EMO版本)Focal_Loss中的alpha参数越大，标签为这一类的loss值越大，为了降低样本数量多的类别的loss影响，需要为样本量多的类别赋小权重
        if each_weighti < num_seen:
            cl_num[each_weighti] = (t1-t2)/t1
        else:
            cl_num[each_weighti] = (t1-cl_num[each_weighti])/t1

    if conf.isFocal_Loss is True:
        criterion = MultiClassFocalLossWithAlpha(alpha=cl_num).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    confu_m = torch.zeros((num_EMO, num_EMO))
    update = UpdateGraph(conf, input_rules).to(device)
    update.train()
    optim_graph = optim.SGD(update.parameters(), lr=init_lr)
    
    infostr = {'init_lr {}'.format(init_lr)}
    logging.info(infostr)
    
    for idx in range(labelsAU.shape[0]):
        torch.cuda.empty_cache()
        adjust_rules_lr_v2(optim_graph, init_lr, idx, train_size)
        cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
        emo_label = labelsEMO[idx].reshape(1,).to(device)

        occ_au = []
        for priori_au_i, priori_au in enumerate(priori_AU):
            if priori_au in dataset_AU:
                pos_priori_in_data = dataset_AU.index(priori_au)
                if cur_item[0, pos_priori_in_data] == 1: 
                    occ_au.append(priori_au_i)
                    # AU_cnt[priori_au_i] += 1
                    if emo_label >= num_seen: 
                        AU_cnt[priori_au_i] += 1

        if len(occ_au) > 1: # len(occ_au) != 0
            num_all_img += 1
            prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU, thresh=0.6) # 计算当前样本中AU的发生概率 P(AU | x)

            cur_prob, _ = update(prob_all_au)
            cur_pred = torch.argmax(cur_prob)
            optim_graph.zero_grad()
            
            if conf.isClass_Weight is True:
                if conf.isClass_Weight_decay is True:
                    cur_err_weight = adjust_loss_weight(loss_weight[emo_label], idx, cl_num[emo_label])
                    err_cls = criterion(cur_prob, emo_label)*cur_err_weight
                else:
                    err_cls = criterion(cur_prob, emo_label)*loss_weight[emo_label]
            else:
                err_cls = criterion(cur_prob, emo_label)

            err = err_cls
            acc = torch.eq(cur_pred, emo_label).sum().item()
            err_record.append(err.item())
            acc_record.append(acc)
            confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
            summary_writer.add_scalar('train_err', np.array(err_record).mean(), idx)
            summary_writer.add_scalar('train_acc', np.array(acc_record).mean(), idx)
            
            err.backward()
            optim_graph.step()
            # if emo_label >= num_seen:
            #     update, AU_ij_cnt, AU_cpt, AU_cnt = crop_EMO2AU(conf, update, occ_au, AU_ij_cnt, AU_cpt, AU_cnt)
            # else:
            #     update = crop_EMO2AU(conf, update)
            del prob_all_au, cur_prob, cur_pred, err, acc
    update, EMO2AU_cpt, prob_AU = final_return(update, EMO, AU, loc1, loc2)

    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return output_rules, output_records, update

def test_rules(conf, update, input_info, input_rules, AU_p_d, summary_writer, confu_m=None):
    priori_AU, dataset_AU = AU_p_d
    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    
    criterion = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    if confu_m is None:
        confu_m = torch.zeros((num_EMO, num_EMO))

    device = conf.device
    loc1 = conf.loc1
    loc2 = conf.loc2

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
                cur_prob, _ = update(prob_all_au)
                cur_pred = torch.argmax(cur_prob)
                confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
                err = criterion(cur_prob, emo_label)
                acc = torch.eq(cur_pred, emo_label).sum().item()
                err_record.append(err.item())
                acc_record.append(acc)
                summary_writer.add_scalar('val_err', np.array(err_record).mean(), idx)
                summary_writer.add_scalar('val_acc', np.array(acc_record).mean(), idx)

                del prob_all_au, cur_prob, cur_pred, err, acc
    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    return output_records