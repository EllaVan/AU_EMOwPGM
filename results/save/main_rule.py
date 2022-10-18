# 持续训练与持续泛化
# 从BP4D训练训练好的规则开始，逐渐加入RAF、AffectNet、DISFA等数据集，目标是希望在所有的数据上都具备一定的正确规则泛化性
# 加入的数据应该按照一定的顺序，从和BP4D最相近的开始增加，再到规则不太相似的，这样或许能够保证结果更好的可说明性
import os,inspect
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
need_path = [current_dir, parent_dir, os.path.join(parent_dir,'models')]
sys.path = need_path + sys.path
os.chdir(current_dir)

import os
from re import A, M
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import argparse
from easydict import EasyDict as edict
import yaml

from conf import ensure_dir, set_logger
from models.AU_EMO_BP import UpdateGraph_v2 as UpdateGraph
from models.RadiationAUs import RadiateAUs_v2 as RadiateAUs
from models.rule_model import learn_rules, test_rules
from losses import *
from utils import *

import matplotlib.pyplot as plt

import datetime
import pytz

def parser2dict():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--dataset_order', type=str, default=['RAF-DB', 'BP4D', 'AffectNet', 'DISFA'])
    parser.add_argument('--outdir', type=str, default='save')
    parser.add_argument('-b','--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    
    parser.add_argument('-j', '--num_workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--evaluate', action='store_true', help='evaluation mode')

    # --------------------settings for training-------------------
    parser.add_argument('--manualSeed', type=int, default=None)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_idx', type=int, default=20000)
    parser.add_argument('--AUthresh', type=float, default=0.6)
    parser.add_argument('--zeroPad', type=float, default=1e-5)

    parser.add_argument('--priori_alpha', type=float, default=0.5)

    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)

def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message

def get_config(cfg):
    if cfg.dataset == 'BP4D':
        with open('config/BP4D_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'DISFA':
        with open('config/DISFA_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'predsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'RAF-DB':
        with open('config/RAF_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'RAF-DB-compound':
        with open('config/RAF_compound_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
    elif cfg.dataset == 'AffectNet':
        with open('config/AffectNet_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch1_model_fold0.pth'
    elif cfg.dataset == 'CASME':
        with open('config/CASME_config.yaml', 'r') as f:
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

def main(conf):
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results'
    pre_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    for_all_test = []
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        conf.dataset = dataset_name
        conf = get_config(conf)
        cur_outdir = os.path.join(conf.outdir, dataset_name)
        ensure_dir(cur_outdir, 0)
        torch.cuda.empty_cache()
        pre_path = os.path.join(pre_path1, dataset_name, pre_path2)

        info_source = conf.source_list
        cur_path = conf.file_list

        info_path = os.path.join(pre_path, cur_path)
        info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
        
        train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)
        dataset_EMO = train_loader.dataset.EMO
        dataset_info = infolist(dataset_EMO, dataset_AU)

        all_info = torch.load(info_path, map_location='cpu')#['state_dict']
        train_rules_input = (all_info['train_input_info'][info_source[0]], all_info['train_input_info'][info_source[1]])
        val_rules_input = (all_info['val_input_info'][info_source[0]], all_info['val_input_info'][info_source[1]])

        EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
        ori_size = np.sum(np.array(EMO_img_num))
        num_all_img = ori_size
        AU_cnt = prob_AU * ori_size
        AU_ij_cnt = np.zeros_like(AU_cpt)
        for au_ij in range(AU_cpt.shape[0]):
            AU_ij_cnt[:, au_ij] = AU_cpt[:, au_ij] * AU_cnt[au_ij]
        input_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
        
        summary_writer = SummaryWriter(cur_outdir)
        conf.lr_relation = -1
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
        ensure_dir(os.path.join(cur_outdir, info_source_path), 0)
        torch.save(checkpoint, os.path.join(cur_outdir, info_source_path, cur_path))

        infostr_rules = {'Dataset: {} train_rules_loss: {:.5f}, train_rules_acc: {:.2f}, val_rules_loss: {:.5f}, val_rules_acc: {:.2f}'
                        .format(dataset_name, train_rules_loss, 100.* train_rules_acc, val_rules_loss, 100.* val_rules_acc)}
        logging.info(infostr_rules)
        infostr_EMO = {'EMO Rules Train Acc-list:'}
        logging.info(infostr_EMO)
        for i in range(train_confu_m.shape[0]):
            train_confu_m[:, i] = train_confu_m[:, i] / train_confu_m[:, i].sum(axis=0)
        infostr_EMO = dataset_info.info_EMO(torch.diag(train_confu_m).cpu().numpy().tolist())
        logging.info(infostr_EMO)
        infostr_EMO = {'EMO Rules Val Acc-list:'}
        logging.info(infostr_EMO)
        for i in range(val_confu_m.shape[0]):
            val_confu_m[:, i] = val_confu_m[:, i] / val_confu_m[:, i].sum(axis=0)
        infostr_EMO = dataset_info.info_EMO(torch.diag(val_confu_m).cpu().numpy().tolist())
        logging.info(infostr_EMO)
        del train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc     
    
    a = 1

if __name__=='__main__':
    conf = parser2dict()
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)
    # cur_day = str(cur_time).split('.')[0].replace(' ', '_')
    cur_time = str(cur_time).split('.')[0]
    cur_day = cur_time.split(' ')[0]
    cur_clock = cur_time.split(' ')[1]

    global device
    conf.gpu = 1
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)
    ensure_dir(conf.outdir, 0)
    set_logger(conf)
    main(conf)
    a = 1