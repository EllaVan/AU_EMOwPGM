# 持续训练与持续泛化
# 从BP4D训练训练好的规则开始，逐渐加入RAF、AffectNet、DISFA等数据集，目标是希望在所有的数据上都具备一定的正确规则泛化性
# 加入的数据应该按照一定的顺序，从和BP4D最相近的开始增加，再到规则不太相似的，这样或许能够保证结果更好的可说明性

import os
from re import A, M
import logging
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from easydict import EasyDict as edict
import yaml

from conf import ensure_dir
from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs

from utils import *
from tensorboardX import SummaryWriter

# from models.TwoBranch import GraphAU, EAC
# from models.TwoBranch_v2 import GraphAU_SSL as GraphAU
# from models.TwoBranch_v2 import EAC_SSL as EAC
from models import rules_BP4D
from models import rules_DISFA
from losses import *
from utils import *
from conf import set_logger

from rules_learning.utils import plot_confusion_matrix
import matplotlib.pyplot as plt

def parser2dict():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--dataset_order', type=str, default=['BP4D', 'RAF-DB', 'AffectNet', 'DISFA'])
    parser.add_argument('--save_path', type=str, default='save')

    parser.add_argument('-b','--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('-j', '--num_workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--evaluate', action='store_true', help='evaluation mode')

    # --------------------settings for training-------------------
    parser.add_argument('--manualSeed', type=int, default=None)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_idx', type=int, default=20000)
    parser.add_argument('--AUthresh', type=float, default=0.6)
    parser.add_argument('--zeroPad', type=float, default=1e-4)

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
            cfg.file_list = 'epoch8_model_fold0.pth'
            cfg.learn_rules = rules_BP4D.learn_rules
            cfg.test_rules = rules_BP4D.test_rules
    elif cfg.dataset == 'DISFA':
        with open('config/DISFA_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'predsEMO_record']
            cfg.file_list = 'epoch16_model_fold0.pth'
            cfg.learn_rules = rules_DISFA.learn_rules
            cfg.test_rules = rules_DISFA.test_rules
    elif cfg.dataset == 'RAF-DB':
        with open('config/RAF_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch8_model_fold0.pth'
            cfg.learn_rules = rules_BP4D.learn_rules
            cfg.test_rules = rules_BP4D.test_rules
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
            cfg.file_list = 'epoch8_model_fold0.pth'
            cfg.learn_rules = rules_BP4D.learn_rules
            cfg.test_rules = rules_BP4D.test_rules
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
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results0911_theTrust'
    pre_path2 = 'Test/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    for_all_test = []
    checkpoint = {}
    checkpoint['conf.dataset_order'] = conf.dataset_order
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        torch.cuda.empty_cache()
        pre_path = os.path.join(pre_path1, dataset_name, pre_path2)
        conf.dataset = dataset_name
        conf = get_config(conf)

        info_source = conf.source_list
        cur_path = conf.file_list
        learn_rules = conf.learn_rules
        test_rules = conf.test_rules

        info_path = os.path.join(pre_path, cur_path)
        info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
        rules_path = os.path.join(pre_path, info_source_path, cur_path)
        
        train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)

        dataset_EMO = train_loader.dataset.EMO
        dataset_info = infolist(dataset_EMO, dataset_AU)

        if dataset_i == 0:
            conf.outdir = os.path.join(pre_path, 'continuous_v2')
            ensure_dir(conf.outdir, 0)
            set_logger(conf)
            input_rules = torch.load(rules_path, map_location='cpu')['output_rules']
            checkpoint['input_rules'] = input_rules
            num_EMO = len(dataset_EMO)
            all_confu_m = torch.zeros((num_EMO, num_EMO))
        else:
            cur_outdir = os.path.join(conf.outdir, dataset_name)
            ensure_dir(cur_outdir, 0)
            summary_writer = SummaryWriter(cur_outdir)
            num_EMO = len(dataset_EMO)

            all_confu_m = torch.zeros((num_EMO, num_EMO))
        
        all_info = torch.load(info_path, map_location='cpu')#['state_dict']

        train_rules_input = (all_info['train_input_info'][info_source[0]], all_info['train_input_info'][info_source[1]])
        val_rules_input = (all_info['val_input_info'][info_source[0]], all_info['val_input_info'][info_source[1]])
        for_all_test_tmp = (AU_p_d, val_rules_input, test_rules)
        for_all_test.append(for_all_test_tmp)
        conf.lr_relation = 0.005
        lr_relation_shet = [0.005, 0.0005, 0.00005]
        

        if dataset_i > 0:
            conf.lr_relation = lr_relation_shet[dataset_i - 1]
            output_rules, train_records = learn_rules(conf, device, train_rules_input, input_rules, AU_p_d, summary_writer)
            train_rules_loss, train_rules_acc, train_confu_m = train_records
            checkpoint['train_'+dataset_name] = train_records
            checkpoint['rules_'+dataset_name] = output_rules
            test_records = test_rules(conf, device, val_rules_input, output_rules, AU_p_d, summary_writer)
            val_rules_loss, val_rules_acc, val_confu_m = test_records
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

            input_rules = output_rules

        if dataset_i >= 1:
            for AU_p_d, val_rules_input, test_rules in for_all_test:
                temp_summary_path = os.path.join(cur_outdir, 'all_test')
                ensure_dir(temp_summary_path, 0)
                temp_summary_writer = SummaryWriter(temp_summary_path)
                all_test_records = test_rules(conf, device, val_rules_input, output_rules,  AU_p_d, temp_summary_writer,all_confu_m)
                all_rules_loss, all_rules_acc, all_confu_m = all_test_records
                
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

def plot_viz():
    info_path = '/media/data1/wf/AU_EMOwPGM/codes/results0911_theTrust/BP4D/Test/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/continuous/all_done.pth'
    all_info = torch.load(info_path, map_location='cpu')
    cm = all_info['all_test_to_DISFA'][2]
    EMO = all_info['priori_EMO']
    save_path = '/media/data1/wf/AU_EMOwPGM/codes/results0911_theTrust/BP4D/Test/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/continuous/cm.jpg'
    plot_confusion_matrix(cm.detach().cpu().numpy(), EMO, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=save_path)
    a = 1


if __name__=='__main__':
    conf = parser2dict()
    conf.gpu = 1

    global device
    device = torch.device('cuda:{}'.format(conf.gpu))
    main(conf)
    # plot_viz()
    a = 1