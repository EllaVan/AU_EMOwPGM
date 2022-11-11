from easydict import EasyDict as edict
import yaml

import torch
import numpy as np

def read_file():
    path1 = 'extending/save/unseen/ZSL/2022-10-30/BP4D/output.pth'
    file1 = torch.load(path1, map_location='cpu')
    extend_EMO2AU = file1['output_rules'][0]
    path2 = '/media/data1/wf/AU_EMOwPGM/codes/results/save/BP4D/labelsAU_labelsEMO/epoch4_model_fold0.pth'
    file2 = torch.load(path2, map_location='cpu')
    a = file2['output_rules'][0][[0, 1, 3, 4], :]
    b = file2['output_rules'][0][[2, 5], :]
    trained_EMO2AU = np.concatenate([a, b])
    a = 1

def get_config(cfg):
    if cfg.dataset == 'BP4D':
        with open('../config/BP4D_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'DISFA':
        with open('../config/DISFA_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'predsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'RAF-DB':
        with open('../config/RAF_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'RAF-DB-compound':
        with open('../config/RAF_compound_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
    elif cfg.dataset == 'AffectNet':
        with open('../config/AffectNet_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch1_model_fold0.pth'
    elif cfg.dataset == 'CASME':
        with open('../config/CASME_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
    else:
        raise Exception("Unkown Datsets:",cfg.dataset)

    if cfg.fold == 0:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_independent
    else:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_dependent
    cfg.update(datasets_cfg)
    return cfg

# 按照unseen_priori的EMO2AU情况，得到unseen_priori_rule
def get_unseen_priori_rule(train_loader, unseen_loc=[2,5]):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
    unseen_EMO2AU_cpt = EMO2AU_cpt[unseen_loc, :]

    num_AU = len(AU)
    num_unseen = len(unseen_loc)
    AU_cpt_tmp = np.zeros((num_AU, num_AU))  #初始化AU的联合概率表
    for k in range(num_unseen):
        AU_pos_nonzero = np.nonzero(unseen_EMO2AU_cpt[k])[0]  #EMO=k的AU-EMO关联（非0值）的位置
        AU_pos_certain = np.where(unseen_EMO2AU_cpt[k]==1)[0]  #EMO=k的AU-EMO关联（1值）的位置
        for j in range(len(AU_pos_nonzero)):
            if unseen_EMO2AU_cpt[k][AU_pos_nonzero[j]] == 1.0:  #当AUj是确定发生的时候，AUi同时发生的概率就是AUi自己本身的值
                for i in range(len(AU_pos_nonzero)):
                    if i != j:
                        AU_cpt_tmp[AU_pos_nonzero[i]][AU_pos_nonzero[j]] += unseen_EMO2AU_cpt[k][AU_pos_nonzero[i]]
            else:  #而当AUj的发生是不确定的时候，只初始化确定发生的AUi，值为1
                for i in range(len(AU_pos_certain)):
                    AU_cpt_tmp[AU_pos_certain[i]][AU_pos_nonzero[j]] += 1.0
    unseen_AU_cpt = (AU_cpt_tmp / num_unseen) + np.eye(num_AU)
    unseen_prob_AU = np.sum(unseen_EMO2AU_cpt, axis=0) / num_unseen
    EMO_img_num = [230] * num_unseen
    unseen_ori_size = np.sum(np.array(EMO_img_num))
    unseen_num_all_img = unseen_ori_size
    unseen_AU_cnt = unseen_prob_AU * unseen_ori_size
    unseen_AU_ij_cnt = np.zeros_like(unseen_AU_cpt)
    for au_ij in range(unseen_AU_cpt.shape[0]):
        unseen_AU_ij_cnt[:, au_ij] = unseen_AU_cpt[:, au_ij] * unseen_AU_cnt[au_ij]
    EMO = ['fear', 'disgust']
    unseen_rule = unseen_EMO2AU_cpt, unseen_AU_cpt, unseen_prob_AU, unseen_ori_size, unseen_num_all_img, unseen_AU_ij_cnt, unseen_AU_cnt, EMO, AU

    return unseen_rule

# 根据(seen, unseen)得到整体的rule
def get_complete_rule(seen_rules, unseen_rules):
    seen_EMO2AU_cpt, seen_AU_cpt, seen_prob_AU, seen_ori_size, seen_num_all_img, seen_AU_ij_cnt, seen_AU_cnt, seen_EMO, AU = seen_rules
    unseen_EMO2AU_cpt, unseen_AU_cpt, unseen_prob_AU, unseen_ori_size, unseen_num_all_img, unseen_AU_ij_cnt, unseen_AU_cnt, unseen_EMO, AU = unseen_rules
    EMO2AU_cpt = np.concatenate((seen_EMO2AU_cpt, unseen_EMO2AU_cpt))
    EMO = seen_EMO + unseen_EMO

    num_AU = len(AU)
    num_EMO = len(EMO)
    num_seen = seen_EMO2AU_cpt.shape[0]
    num_unseen = unseen_EMO2AU_cpt.shape[0]

    # prob_AU = num_unseen/(num_unseen+num_seen)*unseen_prob_AU + num_seen/(num_unseen+num_seen)*seen_prob_AU
    prob_AU = np.sum(EMO2AU_cpt, axis=0) / num_EMO
    ori_size = seen_ori_size + unseen_ori_size
    num_all_img = seen_num_all_img + unseen_num_all_img

    AU_ij_cnt = seen_AU_ij_cnt + unseen_AU_ij_cnt
    AU_cnt = seen_AU_cnt + unseen_AU_cnt
    AU_cpt = np.zeros((num_AU, num_AU)) + np.eye(num_AU)
    for au_i in range(num_AU):
        for au_j in range(num_AU):
            if au_i != au_j:
                AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]

    # AU_cnt = prob_AU * num_all_img
    # AU_cpt = num_unseen/(num_unseen+num_seen)*unseen_AU_cpt + num_seen/(num_unseen+num_seen)*seen_AU_cpt

    # complete_rule = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    complete_rule = EMO2AU_cpt, seen_AU_cpt, seen_prob_AU, seen_ori_size, seen_num_all_img, seen_AU_ij_cnt, seen_AU_cnt, EMO, AU
    return complete_rule

def rule_order():
    order_name = ['EMO2AU_cpt', 'AU_cpt', 'prob_AU', 'ori_size', 'num_all_img', 'AU_ij_cnt', 'AU_cnt', 'EMO', 'AU']
    order_num = list(range(len(order_name)))
    rule_order = dict(zip(order_name, order_num))
    return rule_order
