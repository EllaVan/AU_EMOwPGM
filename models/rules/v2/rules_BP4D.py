import os
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from conf import ensure_dir

from tensorboardX import SummaryWriter

from models.AU_EMO_BP import UpdateGraph_v2 as UpdateGraph
from models.RadiationAUs import RadiateAUs_v2 as RadiateAUs

from utils import *

def learn_rules(conf, device, input_info, input_rules, AU_p_d, summary_writer, *args):
    lr_relation_flag = 0
    init_lr = conf.lr_relation
    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    train_size = labelsAU.shape[0]

    if args:
        change_weight1 = num_all_img / (num_all_img + train_size)
        change_weight2 = 1
        for changing_item in args:
            change_weight2 = change_weight2 * changing_item
        init_lr = change_weight1 * change_weight2
    criterion = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    confu_m = torch.zeros((num_EMO, num_EMO))

    loc = list(range(EMO2AU_cpt.shape[1]))
    loc1 = loc[:-2]
    loc2 = loc[-2:]
    EMO2AU = EMO2AU_cpt

    update = UpdateGraph(conf, EMO2AU_cpt, prob_AU, loc1, loc2).to(device)
    init_lr = num_all_img / (num_all_img + train_size)
    init_lr = 0.00001
    optim_graph = optim.SGD(update.parameters(), lr=init_lr)
    update.train()

    for idx in range(labelsAU.shape[0]):
        torch.cuda.empty_cache()
        adjust_rules_lr_v2(optim_graph, init_lr, idx, train_size)
        cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
        emo_label = labelsEMO[idx].reshape(1,).to(device)

        occ_au = []
        prob_all_au = np.zeros((len(AU),))
        for i, au in enumerate(AU[:-2]):
            if cur_item[0, i] == 1:
                occ_au.append(i)
                AU_cnt[i] += 1

        if cur_item.sum() != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU, thresh=0.6) # 计算当前样本中AU的发生概率 P(AU | x)

            cur_prob = update(prob_all_au)
            cur_pred = torch.argmax(cur_prob)
            optim_graph.zero_grad()
            err = criterion(cur_prob, emo_label)
            acc = torch.eq(cur_pred, emo_label).sum().item()
            err_record.append(err.item())
            acc_record.append(acc)
            confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
            summary_writer.add_scalar('train_err', np.array(err_record).mean(), idx)
            summary_writer.add_scalar('train_acc', np.array(acc_record).mean(), idx)
            
            err.backward()
            optim_graph.step()

            EMO2AU_cpt = update.EMO2AU_cpt.data.detach().cpu().numpy()
            prob_AU = update.prob_AU.data.detach().cpu().numpy()
            EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
            EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)
            update.EMO2AU_cpt.data.copy_(torch.from_numpy(EMO2AU_cpt))
            for i, au_i in enumerate(occ_au):
                for j, au_j in enumerate(occ_au):
                    if i != j:
                        AU_ij_cnt[au_i][au_j] = AU_ij_cnt[au_i][au_j]+1
                        AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
            for i, j in enumerate(AU[:-2]):
                prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            update.prob_AU.data.copy_(torch.from_numpy(prob_AU))

            del prob_all_au, cur_prob, cur_pred, err, acc
    
    EMO2AU_cpt1 = update.EMO2AU_cpt.data
    EMO2AU_cpt2 = update.static_EMO2AU_cpt.data
    EMO2AU_cpt = torch.cat((EMO2AU_cpt1, EMO2AU_cpt2), dim=1).detach().cpu().numpy()
    prob_AU1 = update.prob_AU.data
    prob_AU2 = update.static_prob_AU.data
    prob_AU = torch.cat((prob_AU1, prob_AU2)).detach().cpu().numpy()

    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return output_rules, output_records, update

def test_rules(conf, update, device, input_info, input_rules, AU_p_d, summary_writer, confu_m=None):

    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    
    criterion = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    if confu_m is None:
        confu_m = torch.zeros((num_EMO, num_EMO))

    loc = list(range(EMO2AU_cpt.shape[1]))
    loc1 = loc[:-2]
    loc2 = loc[-2:]

    # update = UpdateGraph(conf, EMO2AU_cpt, prob_AU, loc1, loc2).to(device)
    update.eval()

    with torch.no_grad():
        for idx in range(labelsAU.shape[0]):
            torch.cuda.empty_cache()
            cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
            emo_label = labelsEMO[idx].reshape(1,).to(device)

            occ_au = []
            prob_all_au = np.zeros((len(AU),))
            for i, au in enumerate(AU[:-2]):
                if cur_item[0, i] == 1:
                    occ_au.append(i)

            if cur_item.sum() != 0:
                prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU_cpt, thresh=0.6)
                cur_prob = update(prob_all_au)
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
    if conf.dataset == 'BP4D':
        if info_source[0][0] == 'p':
            train_f1_AU = all_info['val_input_info']['AU_info']['mean_f1_score']
            change_w = change_w * train_f1_AU
        if info_source[1][0] == 'p':
            train_acc_EMO = all_info['val_input_info']['EMO_info']['acc']
            change_w = change_w * train_acc_EMO
    else:

        if info_source[0][0] == 'p':
            train_f1_AU = conf.train_f1_AU
            change_w = change_w * train_f1_AU
        if info_source[1][0] == 'p':
            train_acc_EMO = all_info['val_input_info']['EMO_info']['acc']
            change_w = change_w * train_acc_EMO

    output_rules, train_records, model = learn_rules(conf, device, train_rules_input, input_rules, AU_p_d, summary_writer)#, change_w)
    train_rules_loss, train_rules_acc, train_confu_m = train_records
    train_info = {}
    train_info['rules_loss'] = train_rules_loss
    train_info['rules_acc'] = train_rules_acc
    train_info['train_confu_m'] = train_confu_m
    val_records = test_rules(conf, model, device, val_rules_input, output_rules, AU_p_d, summary_writer)
    val_rules_loss, val_rules_acc, val_confu_m = val_records
    val_info = {}
    val_info['rules_loss'] = val_rules_loss
    val_info['rules_acc'] = val_rules_acc
    val_info['val_confu_m'] = val_confu_m
    checkpoint = {}
    checkpoint['train_info'] = train_info
    checkpoint['val_info'] = val_info
    checkpoint['output_rules'] = output_rules
    checkpoint['model'] = model
    torch.save(checkpoint, os.path.join(pre_path, info_source_path, cur_path))

    return train_rules_loss, train_rules_acc, train_confu_m, val_rules_loss, val_rules_acc, val_confu_m