import os
from re import A
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging
import pickle as pkl

from models.TwoBranch import GraphAU, EAC
from dataset import *
from losses import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env

from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs
import rules_learning.utils as utils

def learn_rules(input_info, input_rules, lr):
    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules

    for i in range(labelsAU.shape[0]):
        cur_item = labelsAU[i, :].reshape(1, -1)
        emo_label = labelsEMO[i].reshape(1,)
        torch.cuda.empty_cache()
        weight = []
        occ_au = []
        prob_all_au = np.zeros((len(AU),))
        for i, au in enumerate(AU[:-2]):
            if cur_item[0, i] == 1:
                occ_au.append(i)
                prob_all_au[i] = 1
                AU_cnt[i] += 1
            weight.append(EMO2AU_cpt[:, i])
        weight.append(EMO2AU_cpt[:, -2])
        weight.append(EMO2AU_cpt[:, -1])

        if len(occ_au) != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
            pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
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
            init = torch.ones((1, len(EMO)))
            for i in range(weight.shape[1]):
                for j in range(1, 3):
                    init[:, i] *= weight[-j][i]*prob_all_au[-j]
            
            weight = np.where(weight > 0, weight, conf.zeroPad)
            weight = torch.from_numpy(weight).cuda()
            prob_all_au = torch.from_numpy(prob_all_au).cuda()
            update = UpdateGraph(device, in_channels=1, out_channels=len(EMO), W=weight[:-2, :], 
                                prob_all_au=prob_all_au[:-2], init=init).cuda()
            optim_graph = optim.SGD(update.parameters(), lr=lr)
            
            AU_evidence = torch.ones((1, 1)).cuda()
            cur_prob = update(AU_evidence)
            cur_pred = torch.argmax(cur_prob)
            err = nn.CrossEntropyLoss()(cur_prob, emo_label)
            acc = torch.eq(cur_pred, emo_label).type(torch.FloatTensor).item()

            optim_graph.zero_grad()
            err.backward()
            optim_graph.step()
            
            update_info1 = update.fc.weight.grad.cpu().numpy().squeeze()
            update_info2 = update.d1.detach().cpu().numpy().squeeze()
            for emo_i, emo_name in enumerate(EMO):
                for i, j in enumerate(AU[:-2]):
                    factor = update_info2[emo_i] / weight[i, emo_i]
                    weight[i, emo_i] -= lr*update_info1[emo_i]*factor
                    if i in pos:
                        EMO2AU_cpt[emo_i, i] = weight[i, emo_i]
                    else:
                        EMO2AU_cpt[emo_i, i] = 1-weight[i, emo_i]
            EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
            EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)

            for i, au_i in enumerate(occ_au):
                for j, au_j in enumerate(occ_au):
                    if i != j:
                        AU_ij_cnt[au_i][au_j] += 1
                        AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
            for i, j in enumerate(AU[:-2]):
                prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            prob_AU = np.where(prob_AU > 0, prob_AU, conf.zeroPad)
            prob_AU = np.where(prob_AU <= 1, prob_AU, 1)

    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    output_records = (err.item(), acc)
    return output_rules, output_records

def test_rules(input_info, input_rules):
    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules

    for i in range(labelsAU.shape[0]):
        cur_item = labelsAU[i, :].reshape(1, -1)
        emo_label = labelsEMO[i]
        torch.cuda.empty_cache()
        weight = []
        occ_au = []
        prob_all_au = np.zeros((len(AU),))
        for i, au in enumerate(AU[:-2]):
            if cur_item[0, i] == 1:
                occ_au.append(i)
                prob_all_au[i] = 1
            weight.append(EMO2AU_cpt[:, i])
        weight.append(EMO2AU_cpt[:, -2])
        weight.append(EMO2AU_cpt[:, -1])

        if len(occ_au) != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
            pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
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
            init = torch.ones((1, len(EMO)))
            for i in range(weight.shape[1]):
                for j in range(1, 3):
                    init[:, i] *= weight[-j][i]*prob_all_au[-j]
            
            weight = np.where(weight > 0, weight, conf.zeroPad)
            weight = torch.from_numpy(weight).cuda()
            prob_all_au = torch.from_numpy(prob_all_au).cuda()
            update = UpdateGraph(device, in_channels=1, out_channels=len(EMO), W=weight[:-2, :], 
                                prob_all_au=prob_all_au[:-2], init=init).cuda()
            
            AU_evidence = torch.ones((1, 1)).cuda()
            cur_prob = update(AU_evidence)
            cur_pred = torch.argmax(cur_prob)
            err = nn.CrossEntropyLoss()(cur_prob, emo_label)
            acc = torch.eq(cur_pred, emo_label).type(torch.FloatTensor).item()
    output_records = (err.item(), acc)
    return output_records

# Train
def train(conf, net_AU, net_EMO, train_loader, optimizer_AU, optimizer_EMO, epoch, criterion_AU, scheduler_EMO):
    losses_AU = AverageMeter()
    losses_EMO = AverageMeter()
    accs_EMO = AverageMeter()
    losses_rules = AverageMeter()
    accs_rules = AverageMeter()

    net_AU, net_EMO = net_AU.cuda(), net_EMO.cuda()
    net_AU.train()
    net_EMO.train()
    train_loader_len = len(train_loader.dataset)

    lr_relation_flag = 0
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
    ori_size = np.sum(np.array(EMO_img_num))
    num_all_img = ori_size
    AU_ij_cnt = AU_cpt * ori_size
    AU_cnt = prob_AU * ori_size
    AU_evidence = torch.ones((1, 1)).cuda()

    for batch_i, (img1, img2, labelsEMO, labelsAU, index) in enumerate(train_loader):
        torch.cuda.empty_cache()
        if img1 is not None:
            #-------------------------------AU train----------------------------
            adjust_learning_rate(optimizer_AU, epoch, conf.epochs, conf.learning_rate_AU, batch_i, train_loader_len)
            labelsAU = labelsAU.float()
            if torch.cuda.is_available():
                img1, img2, labelsEMO, labelsAU = img1.cuda(), img2.cuda(), labelsEMO.cuda(), labelsAU.cuda()
            optimizer_AU.zero_grad()
            outputs_AU = net_AU(img1)
            loss_AU = criterion_AU(outputs_AU, labelsAU)
            loss_AU.backward()
            optimizer_AU.step()
            losses_AU.update(loss_AU.item(), img1.size(0))
            #-------------------------------AU train----------------------------

            #-------------------------------EMO train----------------------------
            labelsEMO = labelsEMO.reshape(labelsEMO.shape[0])
            output_EMO, hm1 = net_EMO(img1)
            output_EMO_flip, hm2 = net_EMO(img2)
            
            grid_l = generate_flip_grid(conf.w, conf.h, device)
            loss1 = nn.CrossEntropyLoss()(output_EMO, labelsEMO)
            flip_loss_l = ACLoss(hm1, hm2, grid_l, output_EMO)
            loss_EMO = loss1 + conf.lam_EMO * flip_loss_l

            optimizer_EMO.zero_grad()
            loss_EMO.backward()
            optimizer_EMO.step()

            _, predicts_EMO = torch.max(output_EMO, 1)
            correct_num_EMO = torch.eq(predicts_EMO, labelsEMO).sum()
            losses_EMO.update(loss_EMO.item(), 1)
            accs_EMO.update(correct_num_EMO.item()/img1.size(0), 1)
            #-------------------------------EMO train----------------------------

            #-------------------------------rules train----------------------------
            # if epoch == conf.epochs - 1:
            # input_info = (labelsAU, predicts_EMO)
            # input_rules = (EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU)
            # output_rules, output_records = learn_rules(input_info, input_rules, conf.lr_relation)

            lr = conf.lr_relation
            for i in range(labelsAU.shape[0]):
                cur_item = labelsAU[i, :].reshape(1, -1)
                emo_label = labelsEMO[i].reshape(1,)
                torch.cuda.empty_cache()
                weight = []
                occ_au = []
                prob_all_au = np.zeros((len(AU),))
                for i, au in enumerate(AU[:-2]):
                    if cur_item[0, i] == 1:
                        occ_au.append(i)
                        prob_all_au[i] = 1
                        AU_cnt[i] += 1
                    weight.append(EMO2AU_cpt[:, i])
                weight.append(EMO2AU_cpt[:, -2])
                weight.append(EMO2AU_cpt[:, -1])

                if len(occ_au) != 0:
                    num_all_img += 1
                    prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
                    pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
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
                    init = torch.ones((1, len(EMO)))
                    for i in range(weight.shape[1]):
                        for j in range(1, 3):
                            init[:, i] *= weight[-j][i]*prob_all_au[-j]
                    
                    weight = np.where(weight > 0, weight, conf.zeroPad)
                    weight = torch.from_numpy(weight).cuda()
                    prob_all_au = torch.from_numpy(prob_all_au).cuda()
                    update = UpdateGraph(device, in_channels=1, out_channels=len(EMO), W=weight[:-2, :], 
                                        prob_all_au=prob_all_au[:-2], init=init).cuda()
                    optim_graph = optim.SGD(update.parameters(), lr=lr)
                    
                    cur_prob = update(AU_evidence)
                    cur_pred = torch.argmax(cur_prob)
                    err = nn.CrossEntropyLoss()(cur_prob, emo_label)
                    acc = torch.eq(cur_pred, emo_label).type(torch.FloatTensor).item()

                    optim_graph.zero_grad()
                    err.backward()
                    optim_graph.step()
                    
                    update_info1 = update.fc.weight.grad.cpu().numpy().squeeze()
                    update_info2 = update.d1.detach().cpu().numpy().squeeze()
                    for emo_i, emo_name in enumerate(EMO):
                        for i, j in enumerate(AU[:-2]):
                            factor = update_info2[emo_i] / weight[i, emo_i]
                            weight[i, emo_i] -= lr*update_info1[emo_i]*factor
                            if i in pos:
                                EMO2AU_cpt[emo_i, i] = weight[i, emo_i]
                            else:
                                EMO2AU_cpt[emo_i, i] = 1-weight[i, emo_i]
                    EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
                    EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)

                    for i, au_i in enumerate(occ_au):
                        for j, au_j in enumerate(occ_au):
                            if i != j:
                                AU_ij_cnt[au_i][au_j] += 1
                                AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
                    for i, j in enumerate(AU[:-2]):
                        prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
                    prob_AU = np.where(prob_AU > 0, prob_AU, conf.zeroPad)
                    prob_AU = np.where(prob_AU <= 1, prob_AU, 1)
                    losses_rules.update(err.item(), 1)
                    accs_rules.update(acc, 1)
                    if num_all_img-ori_size >=err.item() and lr_relation_flag == 0:
                        lr_relation_flag = 1
                        conf.lr_relation /= 10.0
            output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
            # EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = output_rules
            # err_record, acc_record = output_records
            # losses_rules.update(err_record, 1)
            # accs_rules.update(acc_record/img1.size(0), 1)
            # if num_all_img-ori_size >= conf.lr_decay_idx and lr_relation_flag == 0:
            #     lr_relation_flag = 1
            #     conf.lr_relation /= 10.0
            #-------------------------------rules train----------------------------
        
        scheduler_EMO.step()
        train_info_AU = (losses_AU.avg)
        train_info_EMO = (losses_EMO.avg, accs_EMO.avg)
        train_info_rules = (losses_rules.avg, accs_rules.avg, output_rules)
    return train_info_AU, train_info_EMO, train_info_rules


def val(net_AU, net_EMO, val_loader, criterion_AU, input_rules):
    losses_AU = AverageMeter()
    losses_EMO = AverageMeter()
    accs_EMO = AverageMeter()
    losses_rules = AverageMeter()
    accs_rules = AverageMeter()
    num_EMO = len(val_loader.dataset.EMO)
    confu_m = torch.zeros((num_EMO, num_EMO))

    net_AU, net_EMO = net_AU.cuda(), net_EMO.cuda()
    net_AU.eval()
    net_EMO.eval()
    statistics_list = None
    AU_evidence = torch.ones((1, 1)).cuda()

    for batch_i, (img1, img2, labelsEMO, labelsAU, index) in enumerate(val_loader):
        torch.cuda.empty_cache()
        if img1 is not None:
            with torch.no_grad():
                #-------------------------------AU val----------------------------
                labelsAU = labelsAU.float()
                if torch.cuda.is_available():
                    img1, img2, labelsEMO, labelsAU = img1.cuda(), img2.cuda(), labelsEMO.cuda(), labelsAU.cuda()
                outputs_AU = net_AU(img1)
                loss_AU = criterion_AU(outputs_AU, labelsAU)
                losses_AU.update(loss_AU.item(), img1.size(0))
                update_list = statistics(outputs_AU, labelsAU.detach(), 0.5)
                statistics_list = update_statistics_list(statistics_list, update_list)
                #-------------------------------AU val----------------------------

                #-------------------------------EMO val----------------------------
                labelsEMO = labelsEMO.reshape(labelsEMO.shape[0])
                outputs_EMO, _ = net_EMO(img1)

                loss_EMO = nn.CrossEntropyLoss()(outputs_EMO, labelsEMO)
                _, predicts_EMO = torch.max(outputs_EMO, 1)
                correct_num_EMO = torch.eq(predicts_EMO, labelsEMO).sum()
                losses_EMO.update(loss_EMO.item(), img1.size(0))
                accs_EMO.update(correct_num_EMO.item(), img1.size(0))

                confu_m = confusion_matrix(predicts_EMO, labels=labelsEMO, conf_matrix=confu_m)
                #-------------------------------EMO val----------------------------

                #-------------------------------rules test----------------------------
                # input_info = (labelsAU, predicts_EMO)
                # output_records = test_rules(input_info, input_rules)

                EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
                for i in range(labelsAU.shape[0]):
                    cur_item = labelsAU[i, :].reshape(1, -1)
                    emo_label = labelsEMO[i]
                    torch.cuda.empty_cache()
                    weight = []
                    occ_au = []
                    prob_all_au = np.zeros((len(AU),))
                    for i, au in enumerate(AU[:-2]):
                        if cur_item[0, i] == 1:
                            occ_au.append(i)
                            prob_all_au[i] = 1
                        weight.append(EMO2AU_cpt[:, i])
                    weight.append(EMO2AU_cpt[:, -2])
                    weight.append(EMO2AU_cpt[:, -1])

                    if len(occ_au) != 0:
                        num_all_img += 1
                        prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
                        pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
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
                        init = torch.ones((1, len(EMO)))
                        for i in range(weight.shape[1]):
                            for j in range(1, 3):
                                init[:, i] *= weight[-j][i]*prob_all_au[-j]
                        
                        weight = np.where(weight > 0, weight, conf.zeroPad)
                        weight = torch.from_numpy(weight).cuda()
                        prob_all_au = torch.from_numpy(prob_all_au).cuda()
                        update = UpdateGraph(device, in_channels=1, out_channels=len(EMO), W=weight[:-2, :], 
                                            prob_all_au=prob_all_au[:-2], init=init).cuda()
                        
                        cur_prob = update(AU_evidence)
                        cur_pred = torch.argmax(cur_prob)
                        err = nn.CrossEntropyLoss()(cur_prob, emo_label)
                        acc = torch.eq(cur_pred, emo_label).type(torch.FloatTensor).item()
                        losses_rules.update(err.item(), 1)
                        accs_rules.update(acc, 1)

                # err_record, acc_record = output_records
                # losses_rules.update(err_record, 1)
                # accs_rules.update(acc_record/img1.size(0), 1)
                #-------------------------------rules train----------------------------
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)

    AU_return = (losses_AU.avg, mean_f1_score, f1_score_list, mean_acc, acc_list)
    EMO_return = (losses_EMO.avg, accs_EMO.avg, confu_m)
    rules_return = (losses_rules.avg, accs_rules.avg)
    return AU_return, EMO_return, rules_return


def train_AU(conf,net,train_loader,optimizer,epoch,criterion):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader.dataset)
    for batch_idx, (inputs, img1, labelsEMO, targets, index) in enumerate(train_loader):
        torch.cuda.empty_cache()
        if inputs is not None:
            adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate_AU, batch_idx, train_loader_len)
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), inputs.size(0))
    return losses.avg


def train_EMO(args, model, train_loader,optimizer,scheduler):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    
    model.cuda()
    model.train()

    total_loss = []
    for batch_i, (img1, img2, labels, labelsAU, index) in enumerate(train_loader):
        if img1 is not None:
            img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()
            labels = labels.reshape(labels.shape[0])
            criterion = nn.CrossEntropyLoss(reduction='none')
            output, hm1 = model(img1)
            output_flip, hm2 = model(img2)
            
            grid_l = generate_flip_grid(args.w, args.h, device)
            loss1 = nn.CrossEntropyLoss()(output, labels)
            flip_loss_l = ACLoss(hm1, hm2, grid_l, output)
            loss = loss1 + args.lam_EMO * flip_loss_l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_cnt += 1
            _, predicts = torch.max(output, 1)
            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            running_loss += loss

    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(train_loader.dataset.__len__())
    return acc, running_loss


# Val
def val_AU(net,val_loader,criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, img1, labelsEMO, targets, index) in enumerate(val_loader):
        torch.cuda.empty_cache()
        if inputs is not None:
            with torch.no_grad():
                targets = targets.float()
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                update_list = statistics(outputs, targets.detach(), 0.5)
                statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list


def val_EMO(model, test_loader):
    num_EMO = len(test_loader.dataset.EMO)
    confu_m = torch.zeros((num_EMO, num_EMO))
    with torch.no_grad():
        model.cuda()
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0

        for batch_i, (img1, img2, labels, labelsAU, index) in enumerate(test_loader):
            if img1 is not None:
                img1, labels = img1.cuda(), labels.cuda()
                labels = labels.reshape(labels.shape[0])
                outputs, _ = model(img1)

                loss = nn.CrossEntropyLoss()(outputs, labels)
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts, labels).sum()
                correct_sum += correct_num

                running_loss += loss
                data_num += outputs.size(0)

                confu_m = confusion_matrix(predicts, labels=labels, conf_matrix=confu_m)

        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)
    return test_acc, running_loss, confu_m


def main(conf):
    #------------------------------Data Preparation--------------------------
    # summary_writer = SummaryWriter(conf.outdir)
    train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
    train_weight_AU = train_loader.dataset.train_weight_AU
    EMO = train_loader.dataset.EMO
    AU = train_loader.dataset.AU
    dataset_info = infolist(EMO, AU)
    conf.num_classes = len(AU)
    #------------------------------Data Preparation--------------------------

    #---------------------------------AU Setting-----------------------------
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, test_len))
    net_AU = GraphAU(num_classes=conf.num_classes, neighbor_num=conf.neighbor_num, metric=conf.metric)

    if conf.resume != '': # resume
        logging.info("Resume form | {} ]".format(conf.resume))
        net_AU = load_state_dict(net_AU, conf.resume)
    if torch.cuda.is_available():
        net_AU = nn.DataParallel(net_AU).cuda()
        train_weight_AU = torch.from_numpy(train_weight_AU).cuda()
    criterion_AU = WeightedAsymmetricLoss(weight=train_weight_AU)
    optimizer_AU = optim.AdamW(net_AU.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate_AU, weight_decay=conf.weight_decay)
    #---------------------------------AU Setting-----------------------------

    #---------------------------------EMO Setting-----------------------------
    net_EMO = EAC(conf, num_classes=len(EMO))
    optimizer_EMO = torch.optim.Adam(net_EMO.parameters() , lr=conf.learning_rate_EMO, weight_decay=1e-4)
    scheduler_EMO = torch.optim.lr_scheduler.ExponentialLR(optimizer_EMO, gamma=0.9)
    #---------------------------------EMO Setting-----------------------------

    print('the init learning rate of EMO and AU are ', conf.learning_rate_EMO, conf.learning_rate_AU)

    for epoch in range(conf.start_epoch, conf.epochs):
        lr_AU = optimizer_AU.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr_AU))

        '''
        #---------------------------------AU Part-----------------------------
        train_AU_loss = train_AU(conf,net_AU,train_loader,optimizer_AU,epoch,criterion_AU)
        val_AU_loss, val_AU_mean_f1_score, val_AU_f1_score, val_AU_mean_acc, val_AU_acc = val_AU(net_AU, test_loader, criterion_AU)
        #---------------------------------AU Part-----------------------------

        #---------------------------------EMO Part-----------------------------
        train_EMO_acc, train_EMO_loss = train_EMO(conf, net_EMO, train_loader, optimizer_EMO, scheduler_EMO)
        test_EMO_acc, test_EMO_loss, confuse_EMO = val_EMO(net_EMO, test_loader)
        #---------------------------------EMO Part-----------------------------
        '''
        
        train_info_AU, train_info_EMO, train_info_rules = train(conf, net_AU, net_EMO, train_loader, 
            optimizer_AU, optimizer_EMO, epoch, criterion_AU, scheduler_EMO)
        train_AU_loss = train_info_AU
        train_EMO_loss, train_EMO_acc = train_info_EMO
        train_rules_loss, train_rules_acc, rules = train_info_rules

        val_AU_return, val_EMO_return, val_rules_return = val(net_AU, net_EMO, test_loader, criterion_AU, rules)
        val_AU_loss, val_AU_mean_f1_score, val_AU_f1_score, val_AU_mean_acc, val_AU_acc = val_AU_return
        test_EMO_loss, test_EMO_acc, confuse_EMO = val_EMO_return
        val_rules_loss, val_rules_acc = val_rules_return
        

        #---------------------------------Logging Part--------------------------
        # AUlog
        infostr_AU = {'Epoch: {} train_AU_loss: {:.5f} val_AU_loss: {:.5f} val_AU_mean_f1_score {:.2f} val_AU_mean_acc {:.2f}'
                .format(epoch + 1, train_AU_loss, val_AU_loss, 100.* val_AU_mean_f1_score, 100.* val_AU_mean_acc)}
        logging.info(infostr_AU)
        infostr_AU = {'AU F1-score-list:'}
        logging.info(infostr_AU)
        infostr_AU = dataset_info.info_AU(val_AU_f1_score)
        logging.info(infostr_AU)
        infostr_AU = {'AU Acc-list:'}
        logging.info(infostr_AU)
        infostr_AU = dataset_info.info_AU(val_AU_acc)
        logging.info(infostr_AU)
        '''
        summary_writer.add_scalar('train_AU_loss', train_AU_loss, epoch)
        summary_writer.add_scalar('val_AU_loss', train_AU_loss, epoch)
        summary_writer.add_scalar('val_AU_mean_f1_score', val_AU_mean_f1_score, epoch)
        '''

        # EMOlog
        infostr_EMO = {'Epoch {} train_EMO_acc: {:.5f} train_EMO_loss: {:.5f}  val_EMO_acc: {:.5f} val_EMO_loss: {:.5f}'
                        .format(epoch + 1, train_EMO_acc, train_EMO_loss, test_EMO_acc, test_EMO_loss)}
        logging.info(infostr_EMO)
        infostr_EMO = {'EMO Acc-list:'}
        logging.info(infostr_EMO)
        for i in range(confuse_EMO.shape[0]):
            confuse_EMO[:, i] = confuse_EMO[:, i] / confuse_EMO[:, i].sum(axis=0)
        infostr_EMO = dataset_info.info_EMO(torch.diag(confuse_EMO).cpu().numpy().tolist())
        logging.info(infostr_EMO)
        '''
        summary_writer.add_scalar('train_EMO_loss', train_EMO_loss, epoch)
        summary_writer.add_scalar('train_EMO_acc', train_EMO_acc, epoch)
        summary_writer.add_scalar('test_EMO_loss', test_EMO_loss, epoch)
        summary_writer.add_scalar('test_EMO_acc', test_EMO_acc, epoch)
        '''

        '''
        # RULESlog
        infostr_rules = {'Epoch {} train_rules_acc: {:.5f} train_rules_loss: {:.5f}  val_rules_acc: {:.5f} val_rules_loss: {:.5f}'
                        .format(epoch + 1, train_rules_acc, train_rules_loss, val_rules_acc, val_rules_loss)}
        logging.info(infostr_EMO)
        
        summary_writer.add_scalar('train_rules_loss', train_rules_loss, epoch)
        summary_writer.add_scalar('train_rules_acc', train_rules_acc, epoch)
        summary_writer.add_scalar('val_rules_loss', val_rules_loss, epoch)
        summary_writer.add_scalar('val_rules_acc', val_rules_acc, epoch)
        '''
        #---------------------------------Logging Part--------------------------

        if (epoch+1) % conf.save_epoch == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict_AU': net_AU.state_dict(),
                'state_dict_EMO': net_EMO.state_dict(),
                'optimizer_AU': optimizer_AU.state_dict(),
                'optimizer_EMO': optimizer_EMO.state_dict(),
                'scheduler_EMO': scheduler_EMO.state_dict()
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))
    checkpoint = {
            'epoch': epoch,
            'state_dict_AU': net_AU.state_dict(),
            'state_dict_EMO': net_EMO.state_dict(),
            'optimizer_AU': optimizer_AU.state_dict(),
            'optimizer_EMO': optimizer_EMO.state_dict(),
            'scheduler_EMO': scheduler_EMO.state_dict()
        }
    torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model_fold' + str(conf.fold) + '.pth'))
    shutil.copyfile(os.path.join(conf['outdir'], 'train.log'), os.path.join(conf['outdir'], 'train_copy.log'))

if __name__=='__main__':
    conf = get_config()

    global device
    device = torch.device('cuda:{}'.format(conf.gpu))

    set_env(conf)
    set_outdir(conf) # generate outdir name
    set_logger(conf) # Set the logger
    main(conf)