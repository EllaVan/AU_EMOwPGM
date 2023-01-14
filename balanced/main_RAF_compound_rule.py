'''
balanced 方法训练单个数据集
'''
import os
import sys
current_dir = os.path.dirname(__file__) # 当前文件所属文件夹
parent_dir = os.path.dirname(current_dir) # 当前文件所属父文件夹
need_path = [current_dir, parent_dir]#, os.path.join(parent_dir,'models')]
sys.path = need_path + sys.path
os.chdir(current_dir)

import logging
import shutil
import argparse
from easydict import EasyDict as edict
import yaml
import datetime
import pytz

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from conf import ensure_dir, set_logger
from models.rule_model import learn_rules, test_rules
from utils import *

def parser2dict():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    # parser.add_argument('--dataset_order', type=str, default=['RAF-DB', 'AffectNet', 'BP4D', 'DISFA'])
    parser.add_argument('--dataset_order', type=str, default=['RAF-DB', 'RAF-DB-compound'])
    parser.add_argument('--outdir', type=str, default='save/rule')
    parser.add_argument('-b','--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    
    parser.add_argument('-j', '--num_workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--evaluate', action='store_true', help='evaluation mode')

    # --------------------settings for training-------------------
    parser.add_argument('--manualSeed', type=int, default=0)
    
    parser.add_argument('--lr_decay_idx', type=int, default=20000)
    parser.add_argument('--AUthresh', type=float, default=0.6)
    parser.add_argument('--zeroPad', type=float, default=1e-5)

    # --------------------settings for balanced-------------------
    parser.add_argument('--lr_relation', type=float, default=1e-2)
    parser.add_argument('--isFocal_Loss', type=bool, default=True)
    parser.add_argument('--isClass_Weight', type=bool, default=True)
    parser.add_argument('--isClass_Weight_decay', type=bool, default=True)

    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)

# 根据不同的数据集取不同的数据，同时更新args
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

def get_complete_data(all_info, info_source, bias=0):
    train_inputAU = all_info['train_input_info'][info_source[0]]
    train_inputEMO = all_info['train_input_info'][info_source[1]] + bias
    train_inputAU, train_inputEMO = shuffle_input(train_inputAU, train_inputEMO)
    train_rules_input = (train_inputAU, train_inputEMO)
    val_inputAU = all_info['val_input_info'][info_source[0]]
    val_inputEMO = all_info['val_input_info'][info_source[1]] + bias
    val_rules_input = (val_inputAU, val_inputEMO)
    return train_rules_input, val_rules_input

def main(conf):
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results'
    pre_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    bias_emo = [0, 6]
    train_inputAU = [] 
    train_inputEMO = []
    val_inputAU = [] 
    val_inputEMO = []
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        bias = bias_emo[dataset_i]
        conf.dataset = dataset_name
        conf = get_config(conf)
        cur_outdir = os.path.join(conf.outdir, dataset_name) # 每个数据集的存储位置是conf.outdir/dataset
        ensure_dir(cur_outdir, 0)
        torch.cuda.empty_cache()
        pre_path = os.path.join(pre_path1, dataset_name, pre_path2)

        info_source = conf.source_list # 单个数据集训练好的规则的类别来源（是label还是predict)，其epoch来源应与数据的epoch来源一致
        cur_path = conf.file_list # 数据的epoch来源

        info_path = os.path.join(pre_path, cur_path) # 数据来源路径
        info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
        
        # 获取数据的AU和EMO情况，并设定输出
        train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)
        dataset_EMO = train_loader.dataset.EMO
        dataset_info = infolist(dataset_EMO, dataset_AU)

        # 获取数据
        all_info = torch.load(info_path, map_location='cpu')#['state_dict']
        train_rules_input, val_rules_input = get_complete_data(all_info, info_source, bias)
        train_inputAU.append(train_rules_input[0])
        train_inputEMO.append(train_rules_input[1])
        val_inputAU.append(val_rules_input[0])
        val_inputEMO.append(val_rules_input[1])

        # 获取数据的priori rule
        EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
        ori_size = np.sum(np.array(EMO_img_num))
        num_all_img = ori_size
        AU_cnt = prob_AU * ori_size
        AU_ij_cnt = np.zeros_like(AU_cpt)
        for au_ij in range(AU_cpt.shape[0]):
            AU_ij_cnt[:, au_ij] = AU_cpt[:, au_ij] * AU_cnt[au_ij]
        input_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU

        # input_rules = randomPriori(input_rules) #纯数据驱动，先验知识是随机生成的

    train_inputAU = torch.concat(train_inputAU)
    train_inputEMO = torch.concat(train_inputEMO)
    train_inputAU, train_inputEMO = shuffle_input(train_inputAU, train_inputEMO)
    train_rules_input = (train_inputAU, train_inputEMO)
    val_inputAU = torch.concat(val_inputAU)
    val_inputEMO = torch.concat(val_inputEMO)
    dataset_info = infolist(EMO, dataset_AU)
    val_rules_input = (val_inputAU, val_inputEMO)
    
    # 训练与测试
    summary_writer = SummaryWriter(cur_outdir)
    output_rules, train_records, model = learn_rules(conf, train_rules_input, input_rules, AU_p_d, summary_writer)#, change_w)
    train_rules_loss, train_rules_acc, train_confu_m = train_records
    train_info = {}
    train_info['rules_loss'] = train_rules_loss
    train_info['rules_acc'] = train_rules_acc
    train_info['train_confu_m'] = train_confu_m
    '''
    temp_path = '/media/data1/wf/AU_EMOwPGM/codes/continuous/save/continuous/balanced/2022-11-23/BRA/all_done.pth'
    temp_file = torch.load(temp_path, map_location='cpu')
    output_rules = temp_file['rules_AffectNet']
    from models.AU_EMO_BP import UpdateGraph_continuous as UpdateGraph
    model = UpdateGraph(conf, output_rules).to(device)
    model.eval()
    '''
    val_records = test_rules(conf, model, val_rules_input, output_rules, AU_p_d, summary_writer)
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
    # setup_seed(0)
    conf = parser2dict()
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)
    # cur_day = str(cur_time).split('.')[0].replace(' ', '_')
    cur_time = str(cur_time).split('.')[0]
    cur_day = cur_time.split(' ')[0]
    cur_clock = cur_time.split(' ')[1]

    prefix = 'LR_'+str(conf.lr_relation)
    if conf.isFocal_Loss is True:
        prefix = prefix + '_wFocalLoss'
    else:
        prefix = prefix + '_CELoss'
    if conf.isClass_Weight is True:
        if conf.isClass_Weight_decay is True:
            prefix = prefix + '_wClassWeight_decay'
        else:
            prefix = prefix + '_wClassWeight'
    else:
        prefix = prefix + '_UniformWeight'
    conf.outdir = os.path.join(conf.outdir, cur_day, prefix)

    '''
    conf.outdir = os.path.join('/media/data1/wf/AU_EMOwPGM/codes/continuous/save/continuous/balanced/2022-11-23/BRA', 'temp_conti')
    '''

    global device
    conf.gpu = 1
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)
    ensure_dir(conf.outdir, 0)
    set_logger(conf)
    shutil.copyfile("../models/rule_model.py", os.path.join(conf.outdir,'rule_model.py'))
    shutil.copyfile("./main_rule.py", os.path.join(conf.outdir,'main_rule.py'))
    main(conf)
    a = 1