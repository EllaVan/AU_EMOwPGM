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

seen_nodes = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust']
unseen_nodes = ['embarrassment', 'physical pain']

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

def get_config(cfg):
    if cfg.dataset == 'BP4D_all':
        with open('../../config/BP4D_all_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch1_model_fold0.pth'
    else:
        raise Exception("Unkown Datsets:",cfg.dataset)

    if cfg.fold == 0:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_independent
    else:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_dependent
    cfg.update(datasets_cfg)
    return cfg

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
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results'
    pre_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    rule_path1 = '/media/data1/wf/AU_EMOwPGM/codes/save'
    datasets = ['BP4D_all', ]
    conf = parser2dict()
    parts = [('train', 'train_input_info'), ('test', 'val_input_info')]

    for dataset_name in datasets:
        conf.dataset = dataset_name
        conf = get_config(conf)
        checkpoint = {}
        for stage, part in parts:
            checkpoint[part] = {}

            conf.dataset = dataset_name
            conf = get_config(conf)
            info_source = conf.source_list # 单个数据集训练好的规则的类别来源（是label还是predict)，其epoch来源应与数据的epoch来源一致
            cur_path = conf.file_list # 数据的epoch来源
            
            # 获取数据的AU和EMO情况，并设定输出
            with open(conf.dataset_path, 'rb') as fo:
                pkl_file = pkl.load(fo)

            inputAU = pkl_file[stage][info_source[0].split('_')[0]]
            inputEMO = np.array(pkl_file[stage][info_source[1].split('_')[0]])
            # inputAU = torch.from_numpy(inputAU)
            # inputEMO = torch.from_numpy(np.array(inputEMO))

            fear_loc = list(np.where(inputEMO==6)[0])
            disgust_loc = list(np.where(inputEMO==7)[0])
            unseen_loc = fear_loc + disgust_loc

            fear_AU = inputAU[fear_loc, :]
            fear_EMO = inputEMO[fear_loc]
            disgust_AU = inputAU[disgust_loc, :]
            disgust_EMO = inputEMO[disgust_loc]

            unseen_AU = np.concatenate([fear_AU, disgust_AU])
            unseen_EMO = np.concatenate([fear_EMO, disgust_EMO])
            seen_AU = np.delete(inputAU, unseen_loc, 0)
            seen_EMO = np.delete(inputEMO, unseen_loc, 0)

            # unseen_EMO = resetEMO(unseen_EMO)
            # seen_EMO = resetEMO(seen_EMO)

            seen_AU = torch.from_numpy(seen_AU)
            seen_EMO = torch.from_numpy(seen_EMO)
            unseen_AU = torch.from_numpy(unseen_AU)
            unseen_EMO = torch.from_numpy(unseen_EMO)

            checkpoint[part]['seen_AU'] = seen_AU
            checkpoint[part]['seen_EMO'] = seen_EMO
            checkpoint[part]['unseen_AU'] = unseen_AU
            checkpoint[part]['unseen_EMO'] = unseen_EMO

        seen_priori_rules, seen_trained_rules = get_rules()
        checkpoint['seen_priori_rules'] = seen_priori_rules
        checkpoint['seen_trained_rules'] = seen_trained_rules
        checkpoint['EMO'] = pkl_file['EMO']
        checkpoint['AU'] = pkl_file['AU']

        # unseen_priori_rules = get_unseen_fake_rules(len(unseen_nodes), seen_trained_rules)
        # checkpoint['unseen_priori_rules'] = unseen_priori_rules
        # checkpoint['unseen_priori_stastic_rules'] = pkl_file['com_stastics']

        unseen_stastic_EMO2AU = pkl_file['com_stastic_EMO2AU']
        pad_zero = np.zeros((2, 2))
        unseen_stastic_EMO2AU = np.concatenate((unseen_stastic_EMO2AU, pad_zero), axis=1)
        checkpoint['unseen_stastic_EMO2AU'] = unseen_stastic_EMO2AU

        torch.save(checkpoint, dataset_name+'.pkl')
        a = 1

def get_rules():
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results'
    pre_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    dataset_name = 'BP4D'
    cur_path, info_source = getdata(dataset_name)
    pre_path = os.path.join(pre_path1, dataset_name, pre_path2)
    info_path1 = os.path.join(pre_path, cur_path)
    info_path2 = '/media/data1/wf/AU_EMOwPGM/codes/final_results/v2/BP4D/labelsAU_labelsEMO/epoch4_model_fold0.pth'
    all_info1 = torch.load(info_path1, map_location='cpu')#['state_dict']
    all_info2 = torch.load(info_path2, map_location='cpu')#['state_dict']
    input_rules = all_info1['input_rules']
    output_rules = all_info2['output_rules']
    return input_rules, output_rules

def get_unseen_fake_rules(num_unseen, input_rules):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    fake_EMO2AU_cpt = EMO2AU_cpt.mean(axis=0).reshape(1, -1).repeat(num_unseen, axis=0)
    new_EMO2AU_cpt = np.concatenate([EMO2AU_cpt, fake_EMO2AU_cpt])
    new_EMO = EMO + unseen_nodes
    input_rules = new_EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, new_EMO, AU
    a = 1
    return input_rules

if __name__=='__main__':
    data_split()
    a = 1