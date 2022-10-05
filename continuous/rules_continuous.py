import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import os
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from conf import ensure_dir

from tensorboardX import SummaryWriter

from models.AU_EMO_BP import UpdateGraph_continuous as UpdateGraph
from models.RadiationAUs import RadiateAUs_v2 as RadiateAUs
from utils import *

def crop_EMO2AU(conf, priori_update, occ_au, AU_ij_cnt, AU_cpt, AU_cnt, loc2, EMO):
    EMO2AU_cpt = priori_update.EMO2AU_cpt.data.detach().cpu().numpy()
    prob_AU = priori_update.prob_AU.data.detach().cpu().numpy()
    EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
    EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)
    priori_update.EMO2AU_cpt.data.copy_(torch.from_numpy(EMO2AU_cpt))
    for i, au_i in enumerate(occ_au):
        for j, au_j in enumerate(occ_au):
            if i != j:
                AU_ij_cnt[au_i][au_j] = AU_ij_cnt[au_i][au_j]+1
                AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
    for i, j in enumerate(loc2):
        prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
    priori_update.prob_AU.data.copy_(torch.from_numpy(prob_AU))
    return priori_update, occ_au, AU_ij_cnt, AU_cpt, AU_cnt, loc2, EMO\

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

def learn_rules(conf, input_info, priori_rules, latest_rules, AU_p_d, summary_writer, init_lr=-1, *args):
    device = conf.device
    loc1 = conf.loc1
    loc2 = conf.loc2
    priori_alpha = conf.priori_alpha
    labelsAU, labelsEMO = input_info
    priori_AU, dataset_AU = AU_p_d
    priori_update = UpdateGraph(conf, priori_rules, loc1, loc2).to(device)
    latest_update = UpdateGraph(conf, latest_rules, loc1, loc2).to(device)

    # priori_update = UpdateGraph(conf, latest_rules, loc1, loc2).to(device)

    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = latest_rules
    EMO2AU = EMO2AU_cpt.copy()
    train_size = labelsAU.shape[0]

    if init_lr == -1:
        init_lr = train_size / (num_all_img + train_size)
    if args:
        change_weight1 = train_size / (num_all_img + train_size)
        change_weight2 = 1
        for changing_item in args:
            change_weight2 = change_weight2 * changing_item
        init_lr = change_weight1 * change_weight2

    priori_alpha =  train_size / (num_all_img + train_size)
    
    criterion = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    confu_m = torch.zeros((num_EMO, num_EMO))
    
    optim_graph_priori = optim.SGD(priori_update.parameters(), lr=init_lr)
    optim_graph_latest = optim.SGD(latest_update.parameters(), lr=init_lr)
    priori_update.train()
    latest_update.train()
    # for p in latest_update.parameters():
    #     p.requires_grad = False

    for idx in range(labelsAU.shape[0]):
        torch.cuda.empty_cache()
        adjust_rules_lr_v2(optim_graph_priori, init_lr, idx, train_size)
        adjust_rules_lr_v2(optim_graph_latest, init_lr, idx, train_size)
        cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
        emo_label = labelsEMO[idx].reshape(1,).to(device)

        occ_au = []
        for priori_au_i, priori_au in enumerate(priori_AU):
            if priori_au in dataset_AU:
                pos_priori_in_data = dataset_AU.index(priori_au)
                if cur_item[0, pos_priori_in_data] == 1:
                    occ_au.append(priori_au_i)
                    AU_cnt[priori_au_i] += 1

        if cur_item.sum() != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU, thresh=0.6) # 计算当前样本中AU的发生概率 P(AU | x)

            cur_prob_latest, weight2_latest = latest_update(prob_all_au)
            cur_prob_priori, _ = priori_update(prob_all_au, weight2_latest)
            # cur_prob = priori_alpha * cur_prob_priori + (1-cur_prob_priori) * cur_prob_latest
            cur_prob = (1-priori_alpha) * cur_prob_priori + priori_alpha * cur_prob_latest

            # cur_prob, _ = priori_update(prob_all_au)

            cur_pred = torch.argmax(cur_prob)
            optim_graph_priori.zero_grad()
            optim_graph_latest.zero_grad()
            err = criterion(cur_prob, emo_label)
            acc = torch.eq(cur_pred, emo_label).sum().item()
            err_record.append(err.item())
            acc_record.append(acc)
            confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
            summary_writer.add_scalar('train_err', np.array(err_record).mean(), idx)
            summary_writer.add_scalar('train_acc', np.array(acc_record).mean(), idx)
            
            err.backward()
            optim_graph_priori.step()
            optim_graph_latest.step()

            # EMO2AU_cpt = priori_update.EMO2AU_cpt.data.detach().cpu().numpy()
            # prob_AU = priori_update.prob_AU.data.detach().cpu().numpy()
            # EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
            # EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)
            # priori_update.EMO2AU_cpt.data.copy_(torch.from_numpy(EMO2AU_cpt))
            # for i, au_i in enumerate(occ_au):
            #     for j, au_j in enumerate(occ_au):
            #         if i != j:
            #             AU_ij_cnt[au_i][au_j] = AU_ij_cnt[au_i][au_j]+1
            #             AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
            # for i, j in enumerate(loc2):
            #     prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            # priori_update.prob_AU.data.copy_(torch.from_numpy(prob_AU))

            priori_update, occ_au, AU_ij_cnt, AU_cpt, AU_cnt, loc2, EMO = crop_EMO2AU(conf, priori_update, occ_au, AU_ij_cnt, AU_cpt, AU_cnt, loc2, EMO)
            latest_update, _, _, _, _, _, _ = crop_EMO2AU(conf, latest_update, occ_au, AU_ij_cnt, AU_cpt, AU_cnt, loc2, EMO)

            del prob_all_au, cur_prob, cur_pred, err, acc
    
    # EMO2AU_cpt = np.zeros((len(EMO), len(AU)))
    # EMO2AU_cpt1 = priori_update.EMO2AU_cpt.data.detach().cpu().numpy()
    # EMO2AU_cpt2 = priori_update.static_EMO2AU_cpt.data.detach().cpu().numpy()
    # EMO2AU_cpt[:, loc1] = EMO2AU_cpt1
    # EMO2AU_cpt[:, loc2] = EMO2AU_cpt2
    # prob_AU = np.zeros((len(AU),))
    # prob_AU1 = priori_update.prob_AU.data.detach().cpu().numpy()
    # prob_AU2 = priori_update.static_prob_AU.data.detach().cpu().numpy()
    # prob_AU[loc1] = prob_AU1
    # prob_AU[loc2] = prob_AU2

    priori_update, _, _ = final_return(priori_update, EMO, AU, loc1, loc2)
    latest_update, EMO2AU_cpt, prob_AU = final_return(latest_update, EMO, AU, loc1, loc2)

    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return output_rules, output_records, latest_update

def test_rules(conf, update, device, input_info, input_rules, AU_p_d, summary_writer, confu_m=None):
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

            if cur_item.sum() != 0:
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

def do_continuous(conf, device, cur_path, info_source, AU_p_d):
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
    
    # change_w = 1
    # if conf.dataset == 'BP4D':
    #     if info_source[0][0] == 'p':
    #         train_f1_AU = all_info['val_input_info']['AU_info']['mean_f1_score']
    #         change_w = change_w * train_f1_AU
    #     if info_source[1][0] == 'p':
    #         train_acc_EMO = all_info['val_input_info']['EMO_info']['acc']
    #         change_w = change_w * train_acc_EMO
    # else:

    #     if info_source[0][0] == 'p':
    #         train_f1_AU = conf.train_f1_AU
    #         change_w = change_w * train_f1_AU
    #     if info_source[1][0] == 'p':
    #         train_acc_EMO = all_info['val_input_info']['EMO_info']['acc']
    #         change_w = change_w * train_acc_EMO

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