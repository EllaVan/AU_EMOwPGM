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
    if cfg.dataset == 'BP4D':
        with open('../config/BP4D_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'BP4D_all':
        with open('../config/BP4D_all_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch1_model_fold0.pth'
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
            cfg.file_list = 'epoch1_model_fold0.pth'
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