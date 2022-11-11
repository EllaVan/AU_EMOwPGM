import os,inspect
from queue import PriorityQueue
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

from sklearn import decomposition

import argparse
from easydict import EasyDict as edict
import yaml

from conf import ensure_dir, set_logger
from models.AU_EMO_BP import UpdateGraph_continuous as UpdateGraph
from models.rule_model import learn_rules, test_rules
from rule_extend import proj_func
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
    parser.add_argument('--dataset_order', type=str, default=['BP4D', 'DISFA', 'RAF-DB', 'AffectNet'])
    parser.add_argument('--outdir', type=str, default='save/unseen/ZSL')
    parser.add_argument('--rule_dir', type=str, default='save/seen')
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

def get_rest_rule(train_loader):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
    ori_size = np.sum(np.array(EMO_img_num))
    num_all_img = ori_size
    AU_cnt = prob_AU * ori_size
    # AU_ij_cnt = AU_cpt * ori_size
    AU_ij_cnt = np.zeros_like(AU_cpt)
    for au_ij in range(AU_cpt.shape[0]):
        AU_ij_cnt[:, au_ij] = AU_cpt[:, au_ij] * AU_cnt[au_ij]
    input_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    unseen_loc = [2, 5]
    unseen_EMO2AU_cpt = EMO2AU_cpt[unseen_loc, :]
    return unseen_EMO2AU_cpt

def temp_func(seen_rules, unseen_EMO2AU_cpt):
    seen_EMO2AU_cpt, seen_AU_cpt, seen_prob_AU, seen_ori_size, seen_num_all_img, seen_AU_ij_cnt, seen_AU_cnt, EMO, AU = seen_rules
    num_AU = len(AU)
    num_seen = seen_EMO2AU_cpt.shape[0]
    num_unseen = unseen_EMO2AU_cpt.shape[0]
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

    complete_rule = unseen_EMO2AU_cpt, unseen_AU_cpt, unseen_prob_AU, seen_ori_size, seen_num_all_img, seen_AU_ij_cnt, seen_AU_cnt, EMO, AU
    return complete_rule


def get_complete_rule(seen_rules, unseen_EMO2AU_cpt):
    seen_EMO2AU_cpt, seen_AU_cpt, seen_prob_AU, seen_ori_size, seen_num_all_img, seen_AU_ij_cnt, seen_AU_cnt, EMO, AU = seen_rules
    EMO2AU_cpt = np.concatenate((seen_EMO2AU_cpt, unseen_EMO2AU_cpt))

    num_AU = len(AU)
    num_seen = seen_EMO2AU_cpt.shape[0]
    num_unseen = unseen_EMO2AU_cpt.shape[0]
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

    num_EMO = 6
    prob_AU = np.sum(EMO2AU_cpt, axis=0) / num_EMO
    AU_cpt = num_unseen/(num_unseen+num_seen)*unseen_AU_cpt + num_seen/(num_unseen+num_seen)*seen_AU_cpt
    num_all_img = seen_num_all_img/num_seen*(num_seen+num_unseen)
    AU_cnt = prob_AU * num_all_img
    AU_ij_cnt = np.zeros_like(AU_cpt)
    for au_ij in range(AU_cpt.shape[0]):
        AU_ij_cnt[:, au_ij] = AU_cpt[:, au_ij] * AU_cnt[au_ij]

    complete_rule = EMO2AU_cpt, seen_AU_cpt, prob_AU, seen_ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return complete_rule

def get_gauss_key(EMO2AU_cpt):
    num_AU = EMO2AU_cpt.shape[1]
    AU_gauss_key = []
    for i in range(num_AU):
        mean_key = np.mean(EMO2AU_cpt[:, i])
        sigma_key = np.var(EMO2AU_cpt[:, i])
        AU_gauss_key.append((mean_key, sigma_key))
    return AU_gauss_key


def get_multi_gauss(seen_priori_rules, seen_trained_rules, unseen_priori_EMO2AU_cpt):
    seen_priori_EMO2AU_cpt, seen_priori_AU_cpt, seen_priori_prob_AU, seen_ori_size, seen_num_all_img, seen_priori_AU_ij_cnt, seen_priori_AU_cnt, EMO, AU = seen_priori_rules
    seen_trained_EMO2AU_cpt, seen_trained_AU_cpt, seen_trained_prob_AU, seen_ori_size, seen_num_all_img, seen_trained_AU_ij_cnt, seen_trained_AU_cnt, EMO, AU = seen_trained_rules

    priori_EMO2AU_cpt = np.concatenate((seen_priori_EMO2AU_cpt, unseen_priori_EMO2AU_cpt))
    priori_AU_gauss_key = get_gauss_key(priori_EMO2AU_cpt)

    u, s, v = np.linalg.svd(seen_priori_EMO2AU_cpt, full_matrices=False)#截断式矩阵分解
    inv = np.matmul(v.T * 1 / s, u.T)#求逆矩阵
    unseen2seen_W = np.matmul(unseen_priori_EMO2AU_cpt, inv)
    unseen_proj_EMO2AU_cpt = np.matmul(unseen2seen_W, seen_trained_EMO2AU_cpt)
    unseen_proj_EMO2AU_cpt[:, -2:] = unseen_priori_EMO2AU_cpt[:, -2:]

    return priori_AU_gauss_key, unseen2seen_W, unseen_proj_EMO2AU_cpt

def main(conf):
    num_seen = 4
    num_unseen = 2
    pre_data_path = 'dataset'
    seen_rule_path = 'save/seen'
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        conf.dataset = dataset_name
        conf = get_config(conf)
        cur_outdir = os.path.join(conf.outdir, dataset_name)
        ensure_dir(cur_outdir, 0)
        torch.cuda.empty_cache()

        data_path = os.path.join(pre_data_path, dataset_name+'.pkl')
        data_info = torch.load(data_path, map_location='cpu')#['state_dict']
        val_inputAU = torch.cat((data_info['val_input_info']['seen_AU'], data_info['val_input_info']['unseen_AU']))
        val_inputEMO = torch.cat((data_info['val_input_info']['seen_EMO'], data_info['val_input_info']['unseen_EMO']+num_seen))
        val_rules_input = (val_inputAU, val_inputEMO)
        # val_rules_input = (data_info['val_input_info']['seen_AU'], data_info['val_input_info']['seen_EMO'])
        # val_rules_input = (data_info['val_input_info']['unseen_AU'], data_info['val_input_info']['unseen_EMO'])
        
        train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
        unseen_priori_EMO2AU_cpt = get_rest_rule(train_loader)
        seen_priori_rules = data_info['val_input_info']['input_rules']
        seen_rule_info = torch.load(os.path.join(seen_rule_path, dataset_name, 'output.pth'), map_location='cpu')
        seen_trained_rules = seen_rule_info['output_rules']
        priori_AU_gauss_key, unseen2seen_W, unseen_proj_EMO2AU_cpt = get_multi_gauss(seen_priori_rules, seen_trained_rules, unseen_priori_EMO2AU_cpt)
        unseen_proj_EMO2AU_cpt = np.where(unseen_proj_EMO2AU_cpt>0, unseen_proj_EMO2AU_cpt, conf.zeroPad)
        unseen_proj_EMO2AU_cpt = np.where(unseen_proj_EMO2AU_cpt<=1, unseen_proj_EMO2AU_cpt, 1)
        complete_rule = get_complete_rule(seen_trained_rules, unseen_proj_EMO2AU_cpt)
        # complete_rule = temp_func(seen_trained_rules, unseen_proj_EMO2AU_cpt)
        output_rules = complete_rule

        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)
        dataset_EMO = ['happy', 'sad', 'anger', 'surprise', 'fear', 'disgust']
        # dataset_EMO = ['happy', 'sad', 'anger', 'surprise']
        # dataset_EMO = ['fear', 'disgust']
        dataset_info = infolist(dataset_EMO, dataset_AU)
        
        model = UpdateGraph(conf, complete_rule, conf.loc1, conf.loc2).to(device)
        summary_writer = SummaryWriter(cur_outdir)
        val_records = test_rules(conf, model, device, val_rules_input, output_rules, AU_p_d, summary_writer)
        val_rules_loss, val_rules_acc, val_confu_m = val_records
        val_info = {}
        val_info['rules_loss'] = val_rules_loss
        val_info['rules_acc'] = val_rules_acc
        val_info['val_confu_m'] = val_confu_m
        checkpoint = {}
        checkpoint['val_info'] = val_info
        checkpoint['output_rules'] = output_rules
        checkpoint['model'] = model
        torch.save(checkpoint, os.path.join(cur_outdir, 'output.pth'))

        infostr_rules = {'Dataset: {} val_rules_loss: {:.5f}, val_rules_acc: {:.2f}'
                        .format(dataset_name, val_rules_loss, 100.* val_rules_acc)}
        logging.info(infostr_rules)
        infostr_EMO = {'EMO Rules Val Acc-list:'}
        logging.info(infostr_EMO)
        for i in range(val_confu_m.shape[0]):
            val_confu_m[:, i] = val_confu_m[:, i] / val_confu_m[:, i].sum(axis=0)
        infostr_EMO = dataset_info.info_EMO(torch.diag(val_confu_m).cpu().numpy().tolist())
        logging.info(infostr_EMO)
        del val_rules_loss, val_rules_acc     
        
    a = 1

if __name__=='__main__':
    setup_seed(0)
    conf = parser2dict()
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)
    # cur_day = str(cur_time).split('.')[0].replace(' ', '_')
    cur_time = str(cur_time).split('.')[0]
    cur_day = cur_time.split(' ')[0]
    cur_clock = cur_time.split(' ')[1]
    conf.outdir = os.path.join(conf.outdir, cur_day+'_v2')

    global device
    conf.gpu = 1
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)
    ensure_dir(conf.outdir, 0)
    set_logger(conf)
    main(conf)
    a = 1