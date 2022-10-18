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
from itertools import combinations

from conf import ensure_dir, set_logger
from models.AU_EMO_BP import UpdateGraph_continuous as UpdateGraph
from models.RadiationAUs import RadiateAUs_v2 as RadiateAUs

from rules_continuous import learn_rules, test_rules
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
    # parser.add_argument('--dataset_order', type=str, default=['BP4D', 'RAF-DB', 'AffectNet', 'DISFA'])
    parser.add_argument('--dataset_order', type=str, default=['RAF-DB', 'BP4D', 'AffectNet'])
    parser.add_argument('--save_path', type=str, default='save')
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

def get_fenbu(input_info, AU_p_d, EMO_p_d):
    labelsAU, labelsEMO = input_info
    priori_AU, dataset_AU = AU_p_d
    priori_EMO, dataset_EMO = EMO_p_d
    AU_fenbu = [0] * len(priori_AU)
    interAU_fenbu = np.zeros((len(priori_AU), len(priori_AU)))
    EMO_fenbu = [0] * len(dataset_EMO)
    EMO2AU = np.zeros((len(dataset_EMO), len(priori_AU)))

    for idx in range(labelsAU.shape[0]):
        torch.cuda.empty_cache()
        cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
        emo_label = labelsEMO[idx].reshape(1,).to(device)
        EMO_fenbu[emo_label] = EMO_fenbu[emo_label] + 1

        occ_au = []
        for priori_au_i, priori_au in enumerate(priori_AU):
            if priori_au in dataset_AU:
                pos_priori_in_data = dataset_AU.index(priori_au)
                if cur_item[0, pos_priori_in_data] == 1:
                    occ_au.append(priori_au_i)
                    AU_fenbu[priori_au_i] += 1
                    EMO2AU[emo_label][priori_au_i] = EMO2AU[emo_label][priori_au_i] + 1

        for i, au_i in enumerate(occ_au):
            for j, au_j in enumerate(occ_au):
                if i != j:
                    interAU_fenbu[au_i][au_j] = interAU_fenbu[au_i][au_j]+1

    fenbu_return  = (np.array(AU_fenbu), interAU_fenbu, np.array(EMO_fenbu), EMO2AU)
    return fenbu_return

def main(conf):
    # ensure_dir(conf.outdir, 0)
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/results'
    pre_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    for_all_test = []
    checkpoint = {}
    checkpoint['dataset_order'] = conf.dataset_order
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        torch.cuda.empty_cache()
        checkpoint[dataset_name] = {}
        pre_path = os.path.join(pre_path1, dataset_name, pre_path2)
        conf.dataset = dataset_name
        conf = get_config(conf)

        info_source = conf.source_list
        cur_path = conf.file_list

        info_path = os.path.join(pre_path, cur_path)
        info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
        rules_path = os.path.join(pre_path, info_source_path, cur_path)
        
        train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)
        dataset_EMO = train_loader.dataset.EMO
        priori_EMO = train_loader.dataset.priori['EMO']
        EMO_p_d = (priori_EMO, dataset_EMO)
        dataset_info = infolist(dataset_EMO, dataset_AU)

        all_info = torch.load(info_path, map_location='cpu')#['state_dict']
        train_rules_input = (all_info['train_input_info'][info_source[0]], all_info['train_input_info'][info_source[1]])
        val_rules_input = (all_info['val_input_info'][info_source[0]], all_info['val_input_info'][info_source[1]])
        for_all_test_tmp = (AU_p_d, val_rules_input, conf.loc1, conf.loc2)
        for_all_test.append(for_all_test_tmp)

        train_fenbu_return = get_fenbu(train_rules_input, AU_p_d, EMO_p_d)
        train_AU_fenbu, train_interAU_fenbu, train_EMO_fenbu, train_EMO2AU = train_fenbu_return
        val_fenbu_return = get_fenbu(val_rules_input, AU_p_d, EMO_p_d)
        val_AU_fenbu, val_interAU_fenbu, val_EMO_fenbu, val_EMO2AU = val_fenbu_return
        all_AU_fenbu = train_AU_fenbu + val_AU_fenbu
        all_interAU_fenbu = train_interAU_fenbu + val_interAU_fenbu
        all_EMO_fenbu = train_EMO_fenbu + val_EMO_fenbu
        all_EMO2AU = train_EMO2AU + val_EMO2AU
        all_fenbu_return = (all_AU_fenbu, all_interAU_fenbu, all_EMO_fenbu, all_EMO2AU)
        checkpoint[dataset_name]['train_fenbu_return'] = train_fenbu_return
        checkpoint[dataset_name]['val_fenbu_return'] = val_fenbu_return
        checkpoint[dataset_name]['all_fenbu_return'] = all_fenbu_return
        print('{} done'.format(dataset_name))
    save_path = os.path.join(conf.outdir, 'count.pth')
    torch.save(checkpoint, save_path)
    print('Get fenbu (count) done')

    parts = ['train_fenbu_return', 'val_fenbu_return', 'all_fenbu_return']
    fenbu_file = torch.load('save/fenbu/count.pth', map_location='cpu')
    checkpoint = {}
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        checkpoint[dataset_name] = {}
        fenbu_data = fenbu_file[dataset_name]
        for part in parts:
            checkpoint[dataset_name][part] = {}
            AU_fenbu, interAU_fenbu, EMO_fenbu, EMO2AU_fenbu = fenbu_data[part]
            prob_AU = []
            AU_cpt = np.zeros_like(interAU_fenbu)
            img_num = sum(EMO_fenbu)

            for AU_i in range(len(AU_fenbu)):
                prob_AU.append(AU_fenbu[AU_i] / img_num)
                AU_cpt[:, AU_i] = interAU_fenbu[:, AU_i] / AU_fenbu[AU_i]
            EMO_cpt = [emo_num / img_num for emo_num in EMO_fenbu]

            EMO2AU_cpt = np.zeros_like(EMO2AU_fenbu)
            for EMO_i in range(len(EMO_fenbu)):
                EMO2AU_cpt[EMO_i, :] = EMO2AU_fenbu[EMO_i, :] / EMO_fenbu[EMO_i]

            checkpoint[dataset_name][part]['EMO_cpt'] = EMO_cpt
            checkpoint[dataset_name][part]['EMO_fenbu'] = EMO_fenbu
            checkpoint[dataset_name][part]['EMO2AU_fenbu'] = EMO2AU_fenbu
            checkpoint[dataset_name][part]['EMO2AU_cpt'] = EMO2AU_cpt
            checkpoint[dataset_name][part]['prob_AU'] = prob_AU[:-2]
            checkpoint[dataset_name][part]['AU_fenbu'] = AU_fenbu[:-2]
            checkpoint[dataset_name][part]['AU_cpt'] = AU_cpt[:-2, :-2]
            checkpoint[dataset_name][part]['interAU_fenbu'] = interAU_fenbu[:-2, :-2]

    torch.save(checkpoint, os.path.join(conf.outdir, 'prob.pth'))
    print('Get fenbu_prob done')

    end_flag = 1

def read_fenbu(conf): # AU_fenbu, interAU_fenbu, EMO_fenbu
    info = torch.load(os.path.join(conf.outdir, 'prob.pth'), map_location='cpu')
    zuhe = list(combinations(conf.dataset_order, 2))
    checkpoint = {}
    for dataset1, dataset2 in zuhe:
        a = info[dataset1]['train_fenbu_return']['EMO2AU_cpt']
        b = info[dataset2]['train_fenbu_return']['EMO2AU_cpt']
        cov_list = []
        for i in range(a.shape[0]):
            c = a[i, :]
            d = b[i, :]
            cov_list.append(np.cov(c, d)[0, 1])
        checkpoint[dataset1+'_'+dataset2] = np.array(cov_list).reshape(-1, )
    torch.save(checkpoint, os.path.join(conf.outdir, 'cov_EMO2AU.pth'))
    end_flag = 1


def plot_viz():
    info_path = '/media/data1/wf/AU_EMOwPGM/codes/results/BP4D/Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/continuous/all_done.pth'
    all_info = torch.load(info_path, map_location='cpu')
    cm = all_info['all_test_to_RAF-DB'][2]
    EMO = all_info['priori_EMO']
    save_path = '/media/data1/wf/AU_EMOwPGM/codes/results/BP4D/Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/continuous/cm.jpg'
    plot_confusion_matrix(cm.detach().cpu().numpy(), EMO, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=save_path)
    a = 1

if __name__=='__main__':
    conf = parser2dict()

    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)
    # cur_day = str(cur_time).split('.')[0].replace(' ', '_')
    cur_time = str(cur_time).split('.')[0]
    cur_day = cur_time.split(' ')[0]
    cur_clock = cur_time.split(' ')[1]
    conf.outdir = os.path.join(conf.save_path, 'fenbu')

    global device
    conf.gpu = 2
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)
    # main(conf)
    read_fenbu(conf)
    plot_viz()
    a = 1