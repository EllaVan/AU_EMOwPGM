'''
用unseen sample和KL散度度量seen_trained和当前规则中seen部分的分布
'''

import sys
import math
import numpy as np
from functools import reduce
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from tensorboardX import SummaryWriter

from conf import ensure_dir
from model_extend.utils_extend import crop_EMO2AU, final_return
from models.RadiationAUs import RadiateAUs_v2 as RadiateAUs
from models.focal_loss import MultiClassFocalLossWithAlpha
from utils import *

class proj_func(nn.Module):
    def __init__(self, num_AU):
        super(proj_func, self).__init__()
        self.W1 = nn.Linear(num_AU, num_AU, bias=False)
        self.W1.weight.data.fill_(1/num_AU)

    def forward(self, EMO2AU):
        self.EMO2AU = EMO2AU
        self.out1 = self.W1(self.EMO2AU)
        return self.out1

class UpdateGraph(nn.Module):
    def __init__(self, conf, input_rules, proj_W=None, temp=1):
        super(UpdateGraph, self).__init__()
        self.conf = conf
        self.loc1 = conf.loc1
        self.loc2 = conf.loc2
        self.temp = temp

        self.input_rules = input_rules
        EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules

        prob_AU = torch.from_numpy(prob_AU)
        prob_AU = np.where(prob_AU > 0, prob_AU, conf.zeroPad)
        self.register_buffer('prob_AU', torch.from_numpy(prob_AU[self.loc1]))
        self.register_buffer('static_prob_AU', torch.from_numpy(prob_AU[self.loc2]))

        EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
        neg_EMO2AU_cpt = 1 - EMO2AU_cpt
        neg_EMO2AU_cpt = np.where(neg_EMO2AU_cpt > 0, neg_EMO2AU_cpt, conf.zeroPad)

        self.EMO2AU_cpt = Parameter(Variable(torch.from_numpy(EMO2AU_cpt[:, self.loc1])))
        self.EMO2AU_cpt.requires_grad = True

        self.register_buffer('static_EMO2AU_cpt', torch.from_numpy(EMO2AU_cpt[:, self.loc2]))
        self.register_buffer('neg_static_EMO2AU_cpt', torch.from_numpy(neg_EMO2AU_cpt[:, self.loc2]))

        self.AU_cpt = AU_cpt
        self.num_all_img = num_all_img
        self.AU_ij_cnt = AU_ij_cnt
        self.AU_cnt = AU_cnt
        self.EMO = EMO
        self.AU = AU

        if proj_W is None:
            num_loc1 = len(self.loc1)
            proj_W = torch.zeros((num_loc1, num_loc1)) + 1/num_loc1
            self.proj_W = Parameter(Variable(proj_W)).to(torch.float64)
            self.register_buffer('EMO2AU_cpt_proj', torch.matmul(self.EMO2AU_cpt, self.proj_W))

    def get_mask(self, prob_all_au, EMO2AU):
        occ_pos = torch.where(prob_all_au > 0.6)[0]
        occ_mask1 = torch.zeros_like(prob_all_au).cuda()
        neg_mask1 = torch.ones_like(prob_all_au).cuda()
        occ_mask1[occ_pos, :] = 1
        neg_mask1[occ_pos, :] = 0
        occ_mask2 = occ_mask1.reshape(1, -1).repeat(EMO2AU.shape[0], 1).cuda()
        neg_mask2 = neg_mask1.reshape(1, -1).repeat(EMO2AU.shape[0], 1).cuda()
        return occ_pos, occ_mask1, neg_mask1, occ_mask2, neg_mask2

    def forward(self, prob_all_au, weight2=None, is_pre_train=False):
        loc1 = self.loc1
        loc2 = self.loc2
        conf = self.conf
        
        prob_all_au = np.where(prob_all_au > 0, prob_all_au, conf.zeroPad)
        
        self.neg_EMO2AU_cpt = 1 - self.EMO2AU_cpt
        self.neg_EMO2AU_cpt = torch.where(self.neg_EMO2AU_cpt > 0, self.neg_EMO2AU_cpt, conf.zeroPad)
        self.neg_EMO2AU_cpt = torch.where(self.neg_EMO2AU_cpt <= 1, self.neg_EMO2AU_cpt, 1)
        self.prob_all_au = torch.from_numpy(prob_all_au[loc1, :]).cuda()
        self.static_prob_all_au = torch.from_numpy(prob_all_au[loc2, :]).cuda()

        occ_pos, occ_mask1, neg_mask1, occ_mask2, neg_mask2 = self.get_mask(self.prob_all_au, self.EMO2AU_cpt)
        EMO2AU_weight = occ_mask2 * self.EMO2AU_cpt + neg_mask2 * self.neg_EMO2AU_cpt
        AU_weight = occ_mask1.reshape(self.prob_AU.shape) / self.prob_AU * self.prob_all_au.reshape(self.prob_AU.shape) + neg_mask1.reshape(self.prob_AU.shape) / self.prob_AU
        AU_weight = AU_weight.reshape(1, -1).repeat(EMO2AU_weight.shape[0], 1)
        weight1 = EMO2AU_weight * AU_weight

        self.weight2 = weight2
        if weight2 is None:
            occ_pos, occ_mask1, neg_mask1, occ_mask2, neg_mask2 = self.get_mask(self.static_prob_all_au, self.static_EMO2AU_cpt)
            EMO2AU_weight = occ_mask2 * self.static_EMO2AU_cpt + neg_mask2 * self.neg_static_EMO2AU_cpt
            AU_weight = occ_mask1.reshape(self.static_prob_AU.shape) / self.static_prob_AU + neg_mask1.reshape(self.static_prob_AU.shape) / self.static_prob_AU
            AU_weight = AU_weight.reshape(1, -1).repeat(EMO2AU_weight.shape[0], 1)
            self.weight2 = EMO2AU_weight * AU_weight

        a = []
        for i in range(weight1.shape[0]):
            prob_emo = torch.prod(weight1[i, :]) * torch.prod(self.weight2[i, :])
            a.append(prob_emo.reshape(1, -1))

        if is_pre_train is True: # 只输出unseen的预测概率
            self.out1 = torch.cat(a[-2:]).reshape(-1, 2)
            self.out2 = F.normalize(self.out1, p = 1, dim=1)
            self.out3 = F.softmax(self.out2/self.temp, dim=1)
        else: # 全部类别的预测概率
            self.out1 = torch.cat(a).reshape(-1, weight1.shape[0])
            self.out2 = F.normalize(self.out1, p = 1, dim=1)
            self.out3 = F.softmax(self.out2/self.temp, dim=1)

        # 可见类的预测概率
        self.out2_tmp = F.normalize(self.out1[:, :-2], p = 1, dim=1)
        self.out3_tmp = F.softmax(self.out2_tmp/self.temp, dim=1)

        return self.out2, self.out3, self.out2_tmp, self.out3_tmp, self.weight2

def learn_rules(conf, input_info, input_rules, seen_trained_rules, AU_p_d, summary_writer, *args):
    device = conf.device
    loc1 = conf.loc1
    loc2 = conf.loc2
    labelsAU, labelsEMO = input_info
    priori_AU, dataset_AU = AU_p_d

    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    EMO2AU_r = EMO2AU_cpt.copy()
    train_size = labelsAU.shape[0]
    
    seen_trained_EMO2AU_cpt, seen_trained_AU_cpt, seen_trained_prob_AU, seen_trained_ori_size, seen_trained_num_all_img, seen_trained_AU_ij_cnt, seen_trained_AU_cnt, seen_EMO, AU = seen_trained_rules
    num_seen = seen_trained_EMO2AU_cpt.shape[0]
    num_unseen = EMO2AU_cpt.shape[0] - num_seen

    target_seen = seen_trained_EMO2AU_cpt[:, loc1]
    target_seen = torch.from_numpy(target_seen).to(device)
    seen_trained_model = UpdateGraph(conf, seen_trained_rules, temp=2).to(device)
    seen_trained_model.eval()
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
    '''
    for each_weighti in range(len(cl_num)):
        cl_num[each_weighti] = 1-cl_num[each_weighti]/t1 # 此时init_lr=0.01结果还行
    '''
    
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

    criterion = MultiClassFocalLossWithAlpha(alpha=cl_num).to(device)
    # criterion = nn.CrossEntropyLoss()

    acc_record = []
    err_record = []
    cls_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    confu_m = torch.zeros((num_EMO, num_EMO))
    update = UpdateGraph(conf, input_rules, temp=2).to(device)
    optim_graph = optim.SGD(update.parameters(), lr=init_lr)
    
    infostr = {'init_lr {}'.format(init_lr)}
    logging.info(infostr)
    update.train()
    generate_flag = False
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
            prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU_r, thresh=0.6) # 计算当前样本中AU的发生概率 P(AU | x)

            optim_graph.zero_grad()
            
            pre_train_idx = 0.66*len(labelsEMO)
            pre_train_idx = conf.pre_train_idx
            if idx < pre_train_idx:
                # adjust_rules_lr_v2(optim_graph, init_lr, idx, train_size)
                cur_prob, cur_temp_prob, _, _, _ = update(prob_all_au, is_pre_train=True)
                cur_pred = torch.argmax(cur_prob)
                cls_loss = criterion(cur_prob, emo_label-num_seen)
                cls_record.append(cls_loss.item())
                summary_writer.add_scalar('train_cls', np.array(cls_record).mean(), idx)
                err = cls_loss
                acc = torch.eq(cur_pred, emo_label-4).sum().item()
                confu_m = confusion_matrix((cur_pred+4).data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
            else:
                # keep_lr(optim_graph, init_lr)
                cur_prob, cur_temp_prob, cur_seen_prob, cur_seen_temp_prob, _ = update(prob_all_au)
                cur_pred = torch.argmax(cur_prob)

                seen_trained_prob, seen_trained_temp_prob, _, _, _ = seen_trained_model(prob_all_au)
                # seen_trained_temp_prob = torch.zeros_like(seen_trained_prob)
                # KL_loss = cri_KL(update.EMO2AU_cpt[:num_seen, :], target_seen)
                # KL_loss = torch.abs(cri_KL(cur_prob[:, :num_seen], seen_trained_prob))
                # KL_loss =cri_KL(cur_temp_prob[:, :num_seen], seen_trained_temp_prob)
                KL_loss =cri_KL(cur_seen_temp_prob, seen_trained_temp_prob)
                KL_record.append(KL_loss.item())
                summary_writer.add_scalar('train_KL', np.array(KL_record).mean(), idx)

                cls_loss = criterion(cur_prob, emo_label)
                cls_record.append(cls_loss.item())
                summary_writer.add_scalar('train_cls', np.array(cls_record).mean(), idx)
                if emo_label < num_seen:
                    err = cls_loss
                else:
                    err = cls_loss + 0.5 * KL_loss
                # err = cls_loss + 0.5 * KL_loss
                acc = torch.eq(cur_pred, emo_label).sum().item()
                confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)

                # if emo_label >= num_seen:
                #     update, AU_ij_cnt, AU_cpt, AU_cnt = crop_EMO2AU(conf, update, occ_au, AU_ij_cnt, AU_cpt, AU_cnt)
                # else:
                #     update = crop_EMO2AU(conf, update)
            
            err_record.append(err.item())
            acc_record.append(acc)
            summary_writer.add_scalar('train_err', np.array(err_record).mean(), idx)
            summary_writer.add_scalar('train_acc', np.array(acc_record).mean(), idx)
            
            err.backward()
            optim_graph.step()
            update, AU_ij_cnt, AU_cpt, AU_cnt = crop_EMO2AU(conf, update, occ_au, AU_ij_cnt, AU_cpt, AU_cnt)
            del prob_all_au, cur_prob, cur_pred, err, acc

        # if idx > 10000:
            # 下面一直到return 都tab了两次       
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
                prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU_cpt, thresh=0.6, is_interAU=False)
                cur_prob, cur_temp_prob, _, _, _ = update(prob_all_au)#, phase='test')
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