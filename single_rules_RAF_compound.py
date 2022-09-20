import os
from re import A, M
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from conf import parser2dict, ensure_dir, get_config
from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs

from utils import *
from tensorboardX import SummaryWriter

import gc
import objgraph 

def randomPriori(input_rules):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules 
    EMO2AU_cpt = np.random.random(EMO2AU_cpt.shape)
    AU_cpt = np.zeros(AU_cpt.shape)
    for i, j in enumerate(AU[:-2]):
        prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
    ori_size = 0
    num_all_img = ori_size
    AU_ij_cnt = np.zeros(AU_ij_cnt.shape)
    AU_cnt = np.zeros_like(AU_cnt)
    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return output_rules

def learn_rules(conf, device, input_info, input_rules, summary_writer, *args):
    lr_relation_flag = 0
    lr = conf.lr_relation
    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    input_rules = randomPriori(input_rules)

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
            init = np.ones((1, len(EMO)))
            for i in range(weight.shape[1]):
                for j in range(1, 3):
                    init[:, i] = init[:, i]*weight[-j][i]*prob_all_au[-j]
            
            weight = np.where(weight > 0, weight, conf.zeroPad)
            torch.cuda.empty_cache()
            update = UpdateGraph(conf, in_channels=1, out_channels=len(EMO), W=weight[:-2, :], 
                                prob_all_au=prob_all_au[:-2], init=init).to(device)
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
                for i, j in enumerate(AU[:-2]):
                    factor = update_info2[emo_i] / weight[i, emo_i]
                    weight[i, emo_i] = weight[i, emo_i] - update_info1[emo_i]*factor*lr
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
            for i, j in enumerate(AU[:-2]):
                prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            prob_AU = np.where(prob_AU > 0, prob_AU, conf.zeroPad)
            prob_AU = np.where(prob_AU <= 1, prob_AU, 1)
            del cur_item, emo_label, update, optim_graph, cur_prob, cur_pred, err, weight, occ_au, prob_all_au, pos, init, update_info1, update_info2

        # if args is None:
        #     if num_all_img-ori_size >= conf.lr_decay_idx and lr_relation_flag == 0:
        #         lr_relation_flag = 1
        #         lr /= 10.0

    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return output_rules, output_records

def test_rules(conf, device, input_info, input_rules, summary_writer):

    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules

    # EMO2AU_cpt = EMO2AU_cpt[:, :-2]
    # AU_cpt = AU_cpt[:-2, :-2]
    # prob_AU = prob_AU[:-2]
    
    criterion = nn.CrossEntropyLoss()
    AU_evidence = torch.ones((1, 1)).to(device)
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    confu_m = torch.zeros((num_EMO+1, num_EMO))
    # confu_m = torch.zeros((num_EMO+1, 11))

    init_label_record = []
    init_pred_record = []

    for idx in range(labelsAU.shape[0]):
        torch.cuda.empty_cache()
        cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
        emo_label = labelsEMO[idx].reshape(1,).to(device)
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
            # for i in range(prob_all_au.shape[0]):
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
                    init[:, i] = init[:, i]*weight[-j][i]*prob_all_au[-j]
            
            weight = np.where(weight > 0, weight, conf.zeroPad)
            torch.cuda.empty_cache()
            update = UpdateGraph(conf, in_channels=1, out_channels=len(EMO), W=weight[:-2, :], 
                                prob_all_au=prob_all_au[:-2], init=init).to(device)
            # update = UpdateGraph(conf, in_channels=1, out_channels=len(EMO), W=weight, 
            #                     prob_all_au=prob_all_au, init=init).to(device)
            
            cur_prob = update(AU_evidence)
            cur_pred = torch.argmax(cur_prob)
            if update.init[0][cur_pred] / 36.0 >= 0.1:
                confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
                err = criterion(cur_prob, emo_label)
                acc = torch.eq(cur_pred, emo_label).sum().item()
                err_record.append(err.item())
                acc_record.append(acc)
                summary_writer.add_scalar('val_err', np.array(err_record).mean(), idx)
                summary_writer.add_scalar('val_acc', np.array(acc_record).mean(), idx)
                torch.cuda.empty_cache()
                # init_label_record.append(update.init[0][emo_label] / 36.0)
                # init_pred_record.append(update.init[0][cur_pred] / 36.0)
                del cur_item, emo_label, update, cur_prob, cur_pred, err, occ_au, prob_all_au, pos, weight, init
            else:
                confu_m = confusion_matrix([-1], labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    return output_records


def main_rules(conf, device, cur_path, info_source, AU_p_d):
    pre_path = conf.outdir
    info_path = os.path.join(pre_path, cur_path)

    info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
    rules_summary_path = os.path.join(pre_path, 'randomPriori', info_source_path, cur_path.split('.')[0])
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
        train_acc_EMO = all_info['val_input_info']['EMO_info']['acc']
        change_w = change_w * train_acc_EMO

    output_rules, output_records = learn_rules(conf, device, train_rules_input, input_rules, summary_writer, change_w)
    train_rules_loss, train_rules_acc, train_confu_m = output_records
    output_records = test_rules(conf, device, val_rules_input, output_rules, summary_writer)
    val_rules_loss, val_rules_acc, val_confu_m = output_records
    checkpoint = {}
    checkpoint['output_rules'] = output_rules
    torch.save(checkpoint, os.path.join(pre_path, info_source_path, cur_path))

    return train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc, val_confu_m


def main_cross(conf, device, cur_path, info_source):
    pre_path = conf.outdir
    info_path = os.path.join(pre_path, cur_path)

    info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
    rules_summary_path = os.path.join(pre_path, info_source_path, 'extendEMO_thresh_0.1', cur_path.split('.')[0])
    ensure_dir(rules_summary_path, 0)
    summary_writer = SummaryWriter(rules_summary_path)

    rules_path = '/media/data1/wf/AU_EMOwPGM/codes/results0911/BP4D/Test/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/labelsAU_labelsEMO/epoch8_model_fold0.pth'
    output_rules = torch.load(rules_path, map_location='cpu')['output_rules']
    all_info = torch.load(info_path, map_location='cpu')
    val_rules_input = (all_info['val_input_info'][info_source[0]], all_info['val_input_info'][info_source[1]])
    output_records = test_rules(conf, device, val_rules_input, output_rules, summary_writer)
    val_rules_loss, val_rules_acc, val_confu_m = output_records

    return val_rules_loss, val_rules_acc

def main(conf):
    train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
    dataset_AU = train_loader.dataset.AU
    priori_AU = train_loader.dataset.priori['AU']
    AU_p_d = (priori_AU, dataset_AU)
    source_list = [
        ['predsAU_record', 'labelsEMO_record'],
        # ['predsAU_record', 'predsEMO_record']
        ]
    pre_path = '/media/data1/wf/AU_EMOwPGM/codes/results0911/BP4D/Test/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    conf.outdir = pre_path
    # file_list = walkFile(pre_path)
    file_list = ['epoch20_model_fold0.pth']
    
    for info_source in source_list:
        for f in file_list:
            torch.cuda.empty_cache()
            print('The current info source are %s and %s, the features are from %s '%(info_source[0], info_source[1], f))
            # train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc, val_confu_m = main_rules(conf, device, f, info_source, AU_p_d)
            # print('train_rules_loss: {:.5f}, train_rules_acc: {:.5f}, val_rules_loss: {:.5f},, val_rules_acc: {:.5f},'
            #                         .format(train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc))
            # del train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc
            val_rules_loss, val_rules_acc = main_cross(conf, device, f, info_source)
            print('val_rules_loss: {:.5f}, val_rules_acc: {:.5f},' .format(val_rules_loss, val_rules_acc))
            del val_rules_loss, val_rules_acc

if __name__=='__main__':
    conf = parser2dict()
    conf.dataset = 'BP4D'
    conf = get_config(conf)
    conf.gpu = 3

    global device
    device = torch.device('cuda:{}'.format(0))
    main(conf)
    a = 1