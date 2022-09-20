import os
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from conf import ensure_dir

from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs
from tensorboardX import SummaryWriter
from utils import *

def learn_rules(conf, device, input_info, input_rules, summary_writer, AU_p_d, *args):
    lr_relation_flag = 0
    lr = conf.lr_relation
    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    priori_AU, dataset_AU = AU_p_d
    if args:
        train_size = labelsAU.shape[0]
        change_weight1 = ori_size / (ori_size + train_size)
        change_weight2 = 1
        for changing_item in args:
            change_weight2 = change_weight2 * changing_item
        lr = change_weight1 * change_weight2

    criterion = nn.CrossEntropyLoss()
    AU_evidence = torch.ones((1, 1)).to(device)
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    confu_m = torch.zeros((num_EMO, num_EMO))
    for idx in range(labelsAU.shape[0]):
        torch.cuda.empty_cache()
        lr = adjust_rules_lr(lr, idx, train_size)
        cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
        emo_label = labelsEMO[idx].reshape(1,).to(device)
        weight = []
        occ_au = []
        prob_all_au = np.zeros((len(priori_AU),))
        for priori_au_i, priori_au in enumerate(priori_AU):
            if priori_au in dataset_AU:
                pos_priori_in_data = dataset_AU.index(priori_au)
                if cur_item[0, pos_priori_in_data] == 1:
                    occ_au.append(priori_au_i)
                    prob_all_au[priori_au_i] = 1
                    AU_cnt[priori_au_i] += 1
            weight.append(EMO2AU_cpt[:, priori_au_i])

        if len(occ_au) != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
            pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
            weight = np.array(weight)
            for priori_au_i, priori_au in enumerate(priori_AU):
                if priori_au in dataset_AU:
                    if priori_au_i in pos:
                        prob_all_au[priori_au_i] = prob_all_au[priori_au_i] / prob_AU[priori_au_i]
                    else:
                        prob_all_au[priori_au_i] = 1 / (1-prob_AU[priori_au_i])
                        weight[priori_au_i, :] = 1 - weight[priori_au_i, :]
            prob_all_au[5] = 1 / prob_AU[5]
            prob_all_au[7] = 1 / prob_AU[7]
            prob_all_au[8] = 1 / prob_AU[8]
            prob_all_au[13] = 1 / prob_AU[13]
            prob_all_au[14] = 1 / prob_AU[14]
            if emo_label != 3:
                weight[5, :] = 1 - weight[5, :]
                prob_all_au[5] = 1 / (1-prob_AU[5])
                weight[13, :] = 1 - weight[13, :]
                prob_all_au[13] = 1 / (1-prob_AU[13])
            if emo_label == 0 or emo_label == 1 or emo_label == 1:
                weight[7, :] = 1 - weight[7, :]
                prob_all_au[7] = 1 / (1-prob_AU[7])
            if emo_label != 1:
                weight[8, :] = 1 - weight[8, :]
                prob_all_au[8] = 1 / (1-prob_AU[8])
            if emo_label != 3 and emo_label != 5:
                weight[14, :] = 1 - weight[14, :]
                prob_all_au[14] = 1 / (1-prob_AU[14])
            loc1 = [5, 7, 8, 13, 14]
            loc2 = [0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 15, 16]

            init = np.ones((1, len(EMO)))
            for i in range(weight.shape[1]):
                for j in loc1:
                    init[:, i] = init[:, i]*weight[j][i]*prob_all_au[j]
            
            weight = np.where(weight > 0, weight, conf.zeroPad)
            torch.cuda.empty_cache()
            update = UpdateGraph(conf, in_channels=1, out_channels=len(EMO), W=weight[loc2, :], 
                                prob_all_au=prob_all_au[loc2], init=init).to(device)
            optim_graph = optim.SGD(update.parameters(), lr=lr)
            
            cur_prob = update(AU_evidence)
            cur_pred = torch.argmax(cur_prob)
            err = criterion(cur_prob, emo_label)
            acc = torch.eq(cur_pred, emo_label).sum().item()
            err_record.append(err.item())
            acc_record.append(acc)
            confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
            summary_writer.add_scalar('train_err', np.array(err_record).mean(), idx)
            summary_writer.add_scalar('train_acc', np.array(acc_record).mean(), idx)

            optim_graph.zero_grad()
            err.backward()
            optim_graph.step()
            
            torch.cuda.empty_cache()
            update_info1 = update.fc.weight.grad.cpu().numpy().squeeze()
            update_info2 = update.d1.detach().cpu().numpy().squeeze()
            for emo_i, emo_name in enumerate(EMO):
                for i in loc2:
                    factor = update_info2[emo_i] / weight[i, emo_i]
                    weight[i, emo_i] = weight[i, emo_i]-update_info1[emo_i]*factor*lr
                    if i in pos:
                        EMO2AU_cpt[emo_i, i] = weight[i, emo_i]
                    else:
                        EMO2AU_cpt[emo_i, i] = 1-weight[i, emo_i]
            EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
            EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)

            for i, au_i in enumerate(occ_au):
                for j, au_j in enumerate(occ_au):
                    if i != j:
                        AU_ij_cnt[au_i][au_j] = AU_ij_cnt[au_i][au_j]+1
                        AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
            for i, j in enumerate(loc2):
                prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            prob_AU = np.where(prob_AU > 0, prob_AU, conf.zeroPad)
            prob_AU = np.where(prob_AU <= 1, prob_AU, 1)
            del cur_item, emo_label, update, optim_graph, cur_prob, cur_pred, err, weight, occ_au, prob_all_au, pos, init, update_info1, update_info2

        if args is None:
            if num_all_img-ori_size >= conf.lr_decay_idx and lr_relation_flag == 0:
                lr_relation_flag = 1
                lr /= 10.0

    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return output_rules, output_records

def test_rules(conf, device, input_info, input_rules, summary_writer, AU_p_d):

    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    priori_AU, dataset_AU = AU_p_d
    
    criterion = nn.CrossEntropyLoss()
    AU_evidence = torch.ones((1, 1)).to(device)
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    confu_m = torch.zeros((num_EMO, num_EMO))

    for idx in range(labelsAU.shape[0]):
        torch.cuda.empty_cache()
        cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
        emo_label = labelsEMO[idx].reshape(1,).to(device)
        weight = []
        occ_au = []
        prob_all_au = np.zeros((len(AU),))
        for priori_au_i, priori_au in enumerate(priori_AU):
            if priori_au in dataset_AU:
                pos_priori_in_data = dataset_AU.index(priori_au)
                if cur_item[0, pos_priori_in_data] == 1:
                    occ_au.append(priori_au_i)
                    prob_all_au[priori_au_i] = 1
                    AU_cnt[priori_au_i] += 1
            weight.append(EMO2AU_cpt[:, priori_au_i])

        if len(occ_au) != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
            pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
            weight = np.array(weight)
            for priori_au_i, priori_au in enumerate(priori_AU):
                if priori_au in dataset_AU:
                    if priori_au_i in pos:
                        prob_all_au[priori_au_i] = prob_all_au[priori_au_i] / prob_AU[priori_au_i]
                    else:
                        prob_all_au[priori_au_i] = 1 / (1-prob_AU[priori_au_i])
                        weight[priori_au_i, :] = 1 - weight[priori_au_i, :]
            prob_all_au[5] = 1 / prob_AU[5]
            prob_all_au[7] = 1 / prob_AU[7]
            prob_all_au[8] = 1 / prob_AU[8]
            prob_all_au[13] = 1 / prob_AU[13]
            prob_all_au[14] = 1 / prob_AU[14]
            if emo_label != 3:
                weight[5, :] = 1 - weight[5, :]
                prob_all_au[5] = 1 / (1-prob_AU[5])
                weight[13, :] = 1 - weight[13, :]
                prob_all_au[13] = 1 / (1-prob_AU[13])
            if emo_label == 0 or emo_label == 1 or emo_label == 1:
                weight[7, :] = 1 - weight[7, :]
                prob_all_au[7] = 1 / (1-prob_AU[7])
            if emo_label != 1:
                weight[8, :] = 1 - weight[8, :]
                prob_all_au[8] = 1 / (1-prob_AU[8])
            if emo_label != 3 and emo_label != 5:
                weight[14, :] = 1 - weight[14, :]
                prob_all_au[14] = 1 / (1-prob_AU[14])
            loc1 = [5, 7, 8, 13, 14]
            loc2 = [0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 15, 16]

            init = np.ones((1, len(EMO)))
            for i in range(weight.shape[1]):
                for j in loc1:
                    init[:, i] = init[:, i]*weight[j][i]*prob_all_au[j]
            
            weight = np.where(weight > 0, weight, conf.zeroPad)
            torch.cuda.empty_cache()
            update = UpdateGraph(conf, in_channels=1, out_channels=len(EMO), W=weight[loc2, :], 
                                prob_all_au=prob_all_au[loc2], init=init).to(device)
            
            cur_prob = update(AU_evidence)
            cur_pred = torch.argmax(cur_prob)
            confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
            err = criterion(cur_prob, emo_label)
            acc = torch.eq(cur_pred, emo_label).sum().item()
            err_record.append(err.item())
            acc_record.append(acc)
            summary_writer.add_scalar('val_err', np.array(err_record).mean(), idx)
            summary_writer.add_scalar('val_acc', np.array(acc_record).mean(), idx)
            torch.cuda.empty_cache()
            del cur_item, emo_label, update, cur_prob, cur_pred, err, occ_au, prob_all_au, pos, weight, init
    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    return output_records


def main_rules(conf, device, cur_path, info_source, AU_p_d):
    pre_path = conf.outdir
    info_path = os.path.join(pre_path, cur_path)

    info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
    rules_summary_path = os.path.join(pre_path, info_source_path, cur_path.split('.')[0])
    ensure_dir(rules_summary_path, 0)
    summary_writer = SummaryWriter(rules_summary_path)
    
    all_info = torch.load(info_path, map_location='cpu')#['state_dict']
    input_rules = all_info['input_rules']
    train_rules_input = (all_info['train_input_info'][info_source[0]], all_info['train_input_info'][info_source[1]])
    val_rules_input = (all_info['val_input_info'][info_source[0]], all_info['val_input_info'][info_source[1]])

    change_w = 1
    if info_source[0][0] == 'p':
        train_f1_AU = all_info['val_input_info']['AU_info']['mean_f1_score']
        change_w = change_w * train_f1_AU
    if info_source[1][0] == 'p':
        train_acc_EMO = conf.train_acc_EMO
        change_w = change_w * train_acc_EMO
    
    output_rules, output_records = learn_rules(conf, device, train_rules_input, input_rules, summary_writer, AU_p_d, change_w)
    train_rules_loss, train_rules_acc, train_confu_m = output_records
    train_info = {}
    train_info['rules_loss'] = train_rules_loss
    train_info['rules_acc'] = train_rules_acc
    train_info['train_confu_m'] = train_confu_m
    output_records = test_rules(conf, device, val_rules_input, output_rules, summary_writer, AU_p_d)
    val_rules_loss, val_rules_acc, val_confu_m = output_records
    val_info = {}
    val_info['rules_loss'] = val_rules_loss
    val_info['rules_acc'] = val_rules_acc
    val_info['val_confu_m'] = val_confu_m
    checkpoint = {}
    checkpoint['train_info'] = train_info
    checkpoint['val_info'] = val_info
    checkpoint['output_rules'] = output_rules
    torch.save(checkpoint, os.path.join(pre_path, info_source_path, cur_path))

    return train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc, val_confu_m


