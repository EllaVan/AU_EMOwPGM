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
    if dataset_name == 'RAF-DB-compound':
        cur_path = 'epoch1_model_fold0.pth'
        info_source = ['predsAU_record', 'labelsEMO_record']
    elif dataset_name == 'RAF-DB':
        cur_path = 'epoch4_model_fold0.pth'
        info_source = ['predsAU_record', 'labelsEMO_record']
    return cur_path, info_source

seen_nodes = ['happy', 'sad', 'fear', 'anger', 'surprise', 'disgust']
unseen_nodes = ['Happily Surprised', 'Happily Disgusted', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Sadly Disgusted', 
			  'Fearfully Angry', 'Fearfully Surprised', 'Angrily Surprised', 'Angrily Disgusted', 'Disgustedly Surprised']
'''
1: Happily Surprised
2: Happily Disgusted
3: Sadly Fearful
4: Sadly Angry
5: Sadly Surprised
6: Sadly Disgusted
7: Fearfully Angry
8: Fearfully Surprised
9: Angrily Surprised
10: Angrily Disgusted
11: Disgustedly Surprised
'''

def data_split():
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results'
    pre_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    rule_path1 = '/media/data1/wf/AU_EMOwPGM/codes/save'
    datasets = [('seen', 'RAF-DB'), ('unseen', 'RAF-DB-compound')]
    parts = ['train_input_info', 'val_input_info']
    
    checkpoint = {}
    for part in parts:
        checkpoint[part] = {}
        inputAU = []
        inputEMO = []
        for state, dataset_name in datasets:
            cur_path, info_source = getdata(dataset_name)
            pre_path = os.path.join(pre_path1, dataset_name, pre_path2)
            info_path = os.path.join(pre_path, cur_path)
            all_info = torch.load(info_path, map_location='cpu')#['state_dict']
            if state == 'seen':
                num_seen = len(seen_nodes)
                priori_rules = all_info['input_rules']
                trained_info, trained_rules = get_seen_trained_rules()
                latest_rules = trained_rules
            elif  state == 'unseen':
                num_unseen = len(unseen_nodes)
                priori_rules = get_unseen_fake_rules(num_unseen, latest_rules)
                checkpoint[part][state+'_stastic_rules'] = all_info['com_stastics']
                trained_info = None
                trained_rules = None

            checkpoint[part][state+'_AU'] = all_info[part][info_source[0]]
            checkpoint[part][state+'_EMO'] = all_info[part][info_source[1]]
            checkpoint[part][state+'_priori_rules'] = priori_rules
            checkpoint[part][state+'_trained_rules'] = trained_rules
            checkpoint[part][state+'_trained_info'] = trained_info
        
    torch.save(checkpoint, 'RAF-DB-compound.pkl')

def get_seen_trained_rules():
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/save_balanced/v2'
    dataset_name = 'RAF-DB'
    cur_path, info_source = getdata(dataset_name)
    pre_path2 = info_source[0].split('_')[0]+'_'+info_source[1].split('_')[0]
    info_path = os.path.join(pre_path1, dataset_name, pre_path2, cur_path)
    all_info = torch.load(info_path, map_location='cpu')#['state_dict']
    rules = all_info['output_rules']
    return all_info, rules

def get_unseen_fake_rules(num_unseen, input_rules):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    fake_EMO2AU_cpt = EMO2AU_cpt.mean(axis=0).reshape(1, -1).repeat(num_unseen, axis=0)
    new_EMO2AU_cpt = np.concatenate([EMO2AU_cpt, fake_EMO2AU_cpt])
    new_EMO = EMO + unseen_nodes
    input_rules = new_EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, new_EMO, AU
    a = 1
    return input_rules

data_split()
a = 1