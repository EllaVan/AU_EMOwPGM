import os,inspect
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
need_path = [current_dir, parent_dir, os.path.join(parent_dir,'models')]
sys.path = need_path + sys.path
os.chdir(current_dir)

import torch
import numpy as np
import argparse
from easydict import EasyDict as edict
import yaml

from utils import *

def parser2dict():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='BP4D')
    parser.add_argument('-b','--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('-j', '--num_workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)

def getdata(dataset_name):
    if dataset_name == 'BP4D':
        cur_path = 'epoch4_model_fold0.pth'
        info_source = ['labelsAU_record', 'labelsEMO_record']
    elif dataset_name == 'RAF-DB':
        cur_path = 'epoch4_model_fold0.pth'
        info_source = ['predsAU_record', 'labelsEMO_record']
    elif dataset_name == 'AffectNet':
        cur_path = 'epoch1_model_fold0.pth'
        info_source = ['predsAU_record', 'labelsEMO_record']
    elif dataset_name == 'DISFA':
        cur_path = 'epoch4_model_fold0.pth'
        info_source = ['labelsAU_record', 'predsEMO_record']
    return cur_path, info_source

def resetEMO(inputEMO):
    dict_map = {
        0: 0,
        1: 1,
        2: 0,
        3: 2,
        4: 3,
        5: 1
    }
    outputEMO = inputEMO.copy()
    for i in range(inputEMO.shape[0]):
        outputEMO[i] = dict_map[inputEMO[i]]
    return outputEMO

def data_split():
    seen_priori_rules = get_rules()

    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results'
    pre_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    rule_path1 = '/media/data1/wf/AU_EMOwPGM/codes/save'
    datasets = ['BP4D', 'RAF-DB', 'AffectNet', 'DISFA']
    parts = ['train_input_info', 'val_input_info']

    for dataset_name in datasets:
        checkpoint = {}
        cur_path, info_source = getdata(dataset_name)
        pre_path = os.path.join(pre_path1, dataset_name, pre_path2)
        info_path = os.path.join(pre_path, cur_path)
        all_info = torch.load(info_path, map_location='cpu')#['state_dict']

        all_priori_rules = all_info['input_rules']

        for part in parts:
            checkpoint[part] = {}
            inputAU = all_info[part][info_source[0]].numpy()
            inputEMO = all_info[part][info_source[1]].numpy()

            fear_loc = list(np.where(inputEMO==2)[0])
            disgust_loc = list(np.where(inputEMO==5)[0])
            unseen_loc = fear_loc + disgust_loc

            fear_AU = inputAU[fear_loc, :]
            fear_EMO = inputEMO[fear_loc]
            disgust_AU = inputAU[disgust_loc, :]
            disgust_EMO = inputEMO[disgust_loc]

            unseen_AU = np.concatenate([fear_AU, disgust_AU])
            unseen_EMO = np.concatenate([fear_EMO, disgust_EMO])
            seen_AU = np.delete(inputAU, unseen_loc, 0)
            seen_EMO = np.delete(inputEMO, unseen_loc, 0)

            unseen_EMO = resetEMO(unseen_EMO)
            seen_EMO = resetEMO(seen_EMO)

            seen_AU = torch.from_numpy(seen_AU)
            seen_EMO = torch.from_numpy(seen_EMO)
            unseen_AU = torch.from_numpy(unseen_AU)
            unseen_EMO = torch.from_numpy(unseen_EMO)

            checkpoint[part]['seen_AU'] = seen_AU
            checkpoint[part]['seen_EMO'] = seen_EMO
            checkpoint[part]['unseen_AU'] = unseen_AU
            checkpoint[part]['unseen_EMO'] = unseen_EMO
            checkpoint[part]['seen_priori_rules'] = seen_priori_rules
            checkpoint[part]['all_priori_rules'] = all_priori_rules
        
        torch.save(checkpoint, dataset_name+'.pkl')

def get_rules():
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results'
    pre_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    dataset_name = 'BP4D'
    cur_path, info_source = getdata(dataset_name)
    pre_path = os.path.join(pre_path1, dataset_name, pre_path2)
    info_path = os.path.join(pre_path, cur_path)
    all_info = torch.load(info_path, map_location='cpu')#['state_dict']
    input_rules = all_info['input_rules']

    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules

    seen_EMO2AU_cpt = np.delete(EMO2AU_cpt, [2, 5], 0)
    num_EMO = seen_EMO2AU_cpt.shape[0]
    num_AU = seen_EMO2AU_cpt.shape[1]
    EMO_img_num = [230] * num_EMO
    AU_cpt_tmp = np.zeros((num_AU, num_AU))  #初始化AU的联合概率表
    for k in range(num_EMO):
        AU_pos_nonzero = np.nonzero(seen_EMO2AU_cpt[k])[0]  #EMO=k的AU-EMO关联（非0值）的位置
        AU_pos_certain = np.where(seen_EMO2AU_cpt[k]==1)[0]  #EMO=k的AU-EMO关联（1值）的位置
        for j in range(len(AU_pos_nonzero)):
            if seen_EMO2AU_cpt[k][AU_pos_nonzero[j]] == 1.0:  #当AUj是确定发生的时候，AUi同时发生的概率就是AUi自己本身的值
                for i in range(len(AU_pos_nonzero)):
                    if i != j:
                        AU_cpt_tmp[AU_pos_nonzero[i]][AU_pos_nonzero[j]] += seen_EMO2AU_cpt[k][AU_pos_nonzero[i]]
            else:  #而当AUj的发生是不确定的时候，只初始化确定发生的AUi，值为1
                for i in range(len(AU_pos_certain)):
                    AU_cpt_tmp[AU_pos_certain[i]][AU_pos_nonzero[j]] += 1.0
    
    AU_cpt = (AU_cpt_tmp / num_EMO) + np.eye(num_AU)
    EMO2AU_cpt = seen_EMO2AU_cpt
    prob_AU = np.sum(EMO2AU_cpt, axis=0) / num_EMO
    ori_size = np.sum(np.array(EMO_img_num))
    AU_cnt = prob_AU * ori_size
    AU_ij_cnt = np.zeros_like(AU_cpt)
    for au_ij in range(AU_cpt.shape[0]):
        AU_ij_cnt[:, au_ij] = AU_cpt[:, au_ij] * AU_cnt[au_ij]
    EMO = ['happy', 'sad', 'anger', 'surprise']
    input_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU

    return input_rules

data_split()
a = 1