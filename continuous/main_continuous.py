'''
持续训练与持续泛化
从BP4D训练训练好的规则开始, 逐渐加入RAF、AffectNet、DISFA等数据集, 目标是希望在所有的数据上都具备一定的正确规则泛化性
'''
import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
need_path = [current_dir, parent_dir, os.path.join(parent_dir,'models')]
sys.path = need_path + sys.path
os.chdir(current_dir)

import logging
import shutil
import argparse
from easydict import EasyDict as edict
import yaml
import datetime
import pytz

import scipy
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from conf import ensure_dir, set_logger
from models.AU_EMO_BP import UpdateGraph_continuous as UpdateGraph
from model_continuous.rules_continuous import learn_rules, test_rules
from losses import *
from utils import *

def parser2dict():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--fold', type=int, default=0)
    # parser.add_argument('--dataset_order', type=str, default=['BP4D', 'RAF-DB', 'AffectNet', 'DISFA'])
    parser.add_argument('--dataset_order', type=str, default=['RAF-DB', 'BP4D', 'AffectNet'])
    parser.add_argument('--save_path', type=str, default='save/continuous/balanced')
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

def main(conf):
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results'
    pre_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    rule_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results/save'
    for_all_test = []
    checkpoint = {}
    checkpoint['dataset_order'] = conf.dataset_order
    fenbu_info = torch.load(os.path.join('save/fenbu/prob.pth'), map_location='cpu')
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        torch.cuda.empty_cache()
        pre_path = os.path.join(pre_path1, dataset_name, pre_path2)
        conf.dataset = dataset_name
        conf = get_config(conf)

        info_source = conf.source_list
        cur_path = conf.file_list

        info_path = os.path.join(pre_path, cur_path)
        info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
        # rules_path = os.path.join(pre_path, info_source_path, cur_path)
        rules_path = os.path.join(rule_path1, dataset_name, info_source_path, cur_path)
        
        train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)

        dataset_EMO = train_loader.dataset.EMO
        dataset_info = infolist(dataset_EMO, dataset_AU)

        all_info = torch.load(info_path, map_location='cpu')#['state_dict']
        train_rules_input = (all_info['train_input_info'][info_source[0]], all_info['train_input_info'][info_source[1]])
        val_rules_input = (all_info['val_input_info'][info_source[0]], all_info['val_input_info'][info_source[1]])
        for_all_test_tmp = (AU_p_d, val_rules_input, conf.loc1, conf.loc2)
        for_all_test.append(for_all_test_tmp)
        # conf.lr_relation = 0.005
        lr_relation_shet = [0.005, 0.0005, 0.00005]
        # change_w_sheet = [0.1, 0.01] # 0.005
        # change_w_sheet = [0.00167717, 0.06730331] # 不同数据集EMO分布的协方差

        if dataset_i == 0:
            # conf.outdir = os.path.join(pre_path, 'continuous_v3')
            ensure_dir(conf.outdir, 0)
            set_logger(conf)
            # priori_rules = all_info['input_rules']

            EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
            ori_size = np.sum(np.array(EMO_img_num))
            num_all_img = ori_size
            AU_cnt = prob_AU * ori_size
            AU_ij_cnt = np.zeros_like(AU_cpt)
            for au_ij in range(AU_cpt.shape[0]):
                AU_ij_cnt[:, au_ij] = AU_cpt[:, au_ij] * AU_cnt[au_ij]
            priori_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU

            checkpoint['priori_rules'] = priori_rules
            latest_rules = torch.load(rules_path, map_location='cpu')['output_rules']
            # latest_rules = torch.load('save/continuous/2022-10-10_v4/all_done.pth', map_location='cpu')['rules_RAF-DB']
            checkpoint['based_rules'] = latest_rules
            num_EMO = len(dataset_EMO)
            all_confu_m = torch.zeros((num_EMO, num_EMO))

            infostr = {'The priori Dataset {}: The training length is {}, The test length is {}'.format(dataset_name, train_len, test_len)}
            logging.info(infostr)

            latest_AU_fenbu = fenbu_info[dataset_name]['train_fenbu_return']['AU_fenbu']
            latest_EMO_fenbu = fenbu_info[dataset_name]['train_fenbu_return']['EMO_fenbu']
        else:
            cur_outdir = os.path.join(conf.outdir, dataset_name)
            ensure_dir(cur_outdir, 0)
            summary_writer = SummaryWriter(cur_outdir)
            num_EMO = len(dataset_EMO)
            all_confu_m = torch.zeros((num_EMO, num_EMO))
            
            infostr = {'Dataset {}: The training length is {}, The test length is {}'.format(dataset_name, train_len, test_len)}
            logging.info(infostr)

            new_AU_fenbu = fenbu_info[dataset_name]['train_fenbu_return']['AU_fenbu']
            new_EMO_fenbu = fenbu_info[dataset_name]['train_fenbu_return']['EMO_fenbu']

        if dataset_i > 0:
            # conf.lr_relation = lr_relation_shet[dataset_i - 1]

            latest_AU_fenbu = latest_AU_fenbu + new_AU_fenbu
            latest_EMO_fenbu = latest_EMO_fenbu + new_EMO_fenbu
            latest_AU_dis = latest_AU_fenbu / sum(latest_AU_fenbu)
            new_AU_dis = new_AU_fenbu / sum(new_AU_fenbu)
            latest_EMO_dis = latest_EMO_fenbu / sum(latest_EMO_fenbu)
            new_EMO_dis = new_EMO_fenbu / sum(new_EMO_fenbu)
            KL_AU = scipy.stats.entropy(latest_AU_dis, new_AU_dis)
            KL_EMO = scipy.stats.entropy(latest_EMO_dis, new_EMO_dis)
            change_w = KL_AU * KL_EMO
            # if conf.dataset_order[0] == 'BP4D':
            #     change_w = KL_AU * KL_EMO
            # else:
            #     change_w = KL_AU

            conf.lr_relation = -1
            output_rules, train_records, model = learn_rules(conf, train_rules_input, priori_rules, latest_rules, AU_p_d, summary_writer, change_w, 0.23) # change_w_sheet[dataset_i - 1]
            train_rules_loss, train_rules_acc, train_confu_m = train_records
            checkpoint['train_'+dataset_name] = train_records
            checkpoint['rules_'+dataset_name] = output_rules
            model = UpdateGraph(conf, output_rules).to(device)
            test_records = test_rules(conf, model, device, val_rules_input, output_rules, AU_p_d, summary_writer)
            val_rules_loss, val_rules_acc, val_confu_m = test_records
            val_confu_m_copy = val_confu_m.clone()
            checkpoint['test_'+dataset_name] = test_records

            infostr_rules = {'ContiDataOrder {} train_rules_loss: {:.5f}, train_rules_acc: {:.2f}, val_rules_loss: {:.5f}, val_rules_acc: {:.2f}'
                                    .format(dataset_i, train_rules_loss, 100.* train_rules_acc, val_rules_loss, 100.* val_rules_acc)}
            logging.info(infostr_rules)
            
            infostr_EMO = {'EMO Rules Val Acc-list:'}
            logging.info(infostr_EMO)
            for i in range(val_confu_m.shape[0]):
                val_confu_m[:, i] = val_confu_m[:, i] / val_confu_m[:, i].sum(axis=0)
            infostr_EMO = dataset_info.info_EMO(torch.diag(val_confu_m).cpu().numpy().tolist())
            logging.info(infostr_EMO)

            latest_rules = output_rules

        if dataset_i >= 1:
            all_confu_m = val_confu_m_copy
            for cur_i, (AU_p_d, val_rules_input, loc1, loc2) in enumerate(for_all_test[:-1]):
                cur_allto_dataset = conf.dataset_order[cur_i]
                temp_summary_path = os.path.join(cur_outdir, 'all_test', cur_allto_dataset)
                ensure_dir(temp_summary_path, 0)
                model = UpdateGraph(conf, latest_rules).to(device)
                temp_summary_writer = SummaryWriter(temp_summary_path)
                all_test_records = test_rules(conf, model, device, val_rules_input, latest_rules, AU_p_d, temp_summary_writer, all_confu_m)
                all_rules_loss, all_rules_acc, all_confu_m = all_test_records
                infostr_rules = {'The fine-tuned rules val acc on {} is {:.2f}' .format(cur_allto_dataset, 100*all_rules_acc)}
                logging.info(infostr_rules)
                
            all_testEMO_list = torch.diag(all_confu_m).cpu().numpy().tolist()
            all_rules_acc = sum(all_testEMO_list) / torch.sum(all_confu_m)
            infostr_rules = {'AllTestToOrder {} all_rules_acc: {:.2f}' .format(dataset_i, 100*all_rules_acc)}
            logging.info(infostr_rules)
            infostr_EMO = {'AllTestEMO Rules Val Acc-list:'}
            logging.info(infostr_EMO)
            for i in range(all_confu_m.shape[0]):
                all_confu_m[:, i] = all_confu_m[:, i] / all_confu_m[:, i].sum(axis=0)
            infostr_EMO = dataset_info.info_EMO(torch.diag(all_confu_m).cpu().numpy().tolist())
            logging.info(infostr_EMO)

            all_test_records = all_rules_loss, all_rules_acc, all_confu_m
            checkpoint['all_test_to_'+dataset_name] = all_test_records
                
    torch.save(checkpoint, os.path.join(conf.outdir, 'all_done.pth'))
    a = 1

def plot_viz():
    info_path = '/media/data1/wf/AU_EMOwPGM/codes/results/BP4D/Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/continuous/all_done.pth'
    all_info = torch.load(info_path, map_location='cpu')
    cm = all_info['all_test_to_RAF-DB'][2]
    EMO = all_info['priori_EMO']
    save_path = '/media/data1/wf/AU_EMOwPGM/codes/results/BP4D/Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/continuous/cm.jpg'
    plot_confusion_matrix(cm.detach().cpu().numpy(), EMO, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=save_path)
    a = 1

def read():
    path = '/media/data1/wf/AU_EMOwPGM/codes/continuous/save/continuous/2022-10-14/all_done.pth'
    info = torch.load(path, map_location='cpu')
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
    conf.dataset_order = ['BP4D', 'RAF-DB', 'AffectNet']
    if conf.dataset_order[0] == 'BP4D':
        prefix = 'BRA'
        conf.gpu = 1
    else:
        prefix = 'RBA'
        conf.gpu = 2
    conf.outdir = os.path.join(conf.save_path, cur_day, prefix)

    global device
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)
    main(conf)
    # plot_viz()
    # tmp()
    
    # read()
    a = 1