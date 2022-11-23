'''
用unseen sample直接训练unseen_priori得到unseen_trained,随后seen_trained与unseen_trained拼接起来
'''
import os,inspect
from queue import PriorityQueue
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
need_path = [current_dir, parent_dir, os.path.join(parent_dir,'models'), os.path.join(parent_dir,'extending')]
sys.path = need_path + sys.path
os.chdir(current_dir)

import datetime 
import pytz
import argparse
from easydict import EasyDict as edict
import yaml
import logging
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from conf import ensure_dir, set_logger
from model_extend.utils_extend import *
from model_extend.rule_extend2 import UpdateGraph
from model_extend.rule_extend2 import learn_rules, test_rules
from losses import *
from utils import *

def parser2dict():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--dataset_order', type=str, default=['BP4D', 'RAF-DB', 'DISFA', 'AffectNet'])
    parser.add_argument('--outdir', type=str, default='save/unseen/CIL')
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

def interEMO_by_interAU(EMO2AU_cpt, AU_cpt):
    num_EMO = EMO2AU_cpt.shape[0]
    singleAU_influence = np.sum(EMO2AU_cpt, axis=0) 
    AU_trans_inference = np.dot(EMO2AU_cpt, AU_cpt)

    interEMO_cpt = np.zeros((num_EMO, num_EMO))
    for i in range(num_EMO):
        tmp_cor = np.multiply(EMO2AU_cpt[i], AU_trans_inference)
        interEMO_cpt[:, i] = np.sum(tmp_cor, axis=1)

    return interEMO_cpt

def trans_rule(source_rule, target_rule):
    source_EMO2AU_cpt, source_AU_cpt, source_prob_AU, source_ori_size, source_num_all_img, source_AU_ij_cnt, source_AU_cnt, source_EMO, source_AU = source_rule
    target_EMO2AU_cpt, target_AU_cpt, target_prob_AU, target_ori_size, target_num_all_img, target_AU_ij_cnt, target_AU_cnt, target_EMO, target_AU = target_rule

    inter_source_EMO_cpt = interEMO_by_interAU(source_EMO2AU_cpt, source_AU_cpt)
    inter_s2t_EMO_cpt = interEMO_by_interAU(source_EMO2AU_cpt, target_AU_cpt)

    u, s, v = np.linalg.svd(inter_s2t_EMO_cpt, full_matrices=False)#截断式矩阵分解
    inv = np.matmul(v.T * 1 / s, u.T)#求逆矩阵
    source2target_W = np.matmul(inter_source_EMO_cpt, inv)

    return source2target_W

def recouple_rule1(conf, seen_trained_rules, unseen_trained_rules, all_cat_trained_rules):
    seen_trained_EMO2AU_cpt, seen_trained_AU_cpt, seen_trained_prob_AU, seen_trained_ori_size, seen_trained_num_all_img, seen_trained_AU_ij_cnt, seen_trained_AU_cnt, seen_trained_EMO, AU = seen_trained_rules
    unseen_trained_EMO2AU_cpt, unseen_trained_AU_cpt, unseen_trained_prob_AU, unseen_trained_ori_size, unseen_trained_num_all_img, unseen_trained_AU_ij_cnt, AU_cnt, unseen_trained_unseen_trained_EMO, AU = unseen_trained_rules
    all_cat_trained_EMO2AU_cpt, all_cat_trained_AU_cpt, all_cat_trained_prob_AU, all_cat_trained_ori_size, all_cat_trained_num_all_img, all_cat_trained_AU_ij_cnt, all_cat_trained_AU_cnt, all_cat_trained_EMO, AU = all_cat_trained_rules

    seen2all_W = trans_rule(seen_trained_rules, all_cat_trained_rules)
    unseen2all_W = trans_rule(unseen_trained_rules, all_cat_trained_rules)

    seen_EMO2AU_cpt = np.matmul(seen2all_W, seen_trained_EMO2AU_cpt)
    unseen_EMO2AU_cpt = np.matmul(unseen2all_W, unseen_trained_EMO2AU_cpt)
    EMO2AU_cpt = np.concatenate((seen_EMO2AU_cpt, unseen_EMO2AU_cpt))
    EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
    EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)
    num_EMO = EMO2AU_cpt.shape[0]
    prob_AU = np.sum(EMO2AU_cpt, axis=0) / num_EMO

    complete_rule = EMO2AU_cpt, all_cat_trained_AU_cpt, prob_AU, all_cat_trained_ori_size, all_cat_trained_num_all_img, all_cat_trained_AU_ij_cnt, all_cat_trained_AU_cnt, all_cat_trained_EMO, AU
    return complete_rule

def recouple_rule2(conf, seen_trained_rules, unseen_trained_rules, all_cat_trained_rules):
    seen_trained_EMO2AU_cpt, seen_trained_AU_cpt, seen_trained_prob_AU, seen_trained_ori_size, seen_trained_num_all_img, seen_trained_AU_ij_cnt, seen_trained_AU_cnt, seen_trained_EMO, AU = seen_trained_rules
    unseen_trained_EMO2AU_cpt, unseen_trained_AU_cpt, unseen_trained_prob_AU, unseen_trained_ori_size, unseen_trained_num_all_img, unseen_trained_AU_ij_cnt, AU_cnt, unseen_trained_unseen_trained_EMO, AU = unseen_trained_rules
    all_cat_trained_EMO2AU_cpt, all_cat_trained_AU_cpt, all_cat_trained_prob_AU, all_cat_trained_ori_size, all_cat_trained_num_all_img, all_cat_trained_AU_ij_cnt, all_cat_trained_AU_cnt, all_cat_trained_EMO, AU = all_cat_trained_rules

    num_seen = seen_trained_EMO2AU_cpt.shape[0]
    num_unseen = unseen_trained_EMO2AU_cpt.shape[0]
    num_EMO = num_seen + num_unseen
    num_AU = seen_trained_EMO2AU_cpt.shape[1]

    seenAU2EMO = np.zeros_like(seen_trained_EMO2AU_cpt.T)
    unseenAU2EMO = np.zeros_like(unseen_trained_EMO2AU_cpt.T)
    for i in range(num_AU):
        seenAU2EMO[i, :] = seen_trained_EMO2AU_cpt[:, i]/seen_trained_prob_AU[i]*num_seen
        unseenAU2EMO[i, :] = unseen_trained_EMO2AU_cpt[:, i]/unseen_trained_prob_AU[i]*num_unseen

    num_all_img = seen_trained_num_all_img + unseen_trained_num_all_img - seen_trained_ori_size - unseen_trained_ori_size
    seen_num_AU = list(seen_trained_prob_AU*seen_trained_num_all_img)
    unseen_num_AU = list(unseen_trained_prob_AU*unseen_trained_num_all_img)
    all_num_AU = []
    for i in range(num_AU):
        all_num_AU.append(seen_num_AU[i]+unseen_num_AU[i])
    tmp = sum(all_num_AU)
    all_prob_AU = np.array([i/tmp for i in all_num_AU])

    all_EMO2AU = np.zeros((num_EMO, num_AU))
    for i in range(num_EMO):
        if i < num_seen:
            all_EMO2AU[i, :] = seenAU2EMO[:, i]*all_prob_AU[i]/num_seen
            all_EMO2AU[i, conf.loc2] = seen_trained_EMO2AU_cpt[i, conf.loc2]
        else:
            all_EMO2AU[i, :] = unseenAU2EMO[:, i-num_seen]*all_prob_AU[i]/num_unseen
            all_EMO2AU[i, conf.loc2] = unseen_trained_EMO2AU_cpt[i-num_seen, conf.loc2]

    prob_AU = np.sum(all_EMO2AU, axis=0) / num_EMO

    complete_rule = all_EMO2AU, all_cat_trained_AU_cpt, all_prob_AU, all_cat_trained_ori_size, num_all_img, all_cat_trained_AU_ij_cnt, all_cat_trained_AU_cnt, all_cat_trained_EMO, AU
    return complete_rule

def main_CatRule(conf):
    num_seen = 4
    num_unseen = 2
    pre_data_path = 'dataset'
    pre_seen_rule_path = 'save/seen/2022-11-12'
    pre_unseen_rule_path = 'save/unseen/CIL/only_unseen/2022-11-11'
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        conf.dataset = dataset_name
        conf = get_config(conf)
        cur_outdir = os.path.join(conf.outdir, dataset_name)
        ensure_dir(cur_outdir)
        summary_writer = SummaryWriter(cur_outdir)
        torch.cuda.empty_cache()

        data_path = os.path.join(pre_data_path, dataset_name+'.pkl')
        data_info = torch.load(data_path, map_location='cpu')
        train_inputAU = data_info['train_input_info']['unseen_AU']
        train_inputEMO = data_info['train_input_info']['unseen_EMO']+num_seen
        a = list(zip(train_inputAU, train_inputEMO))
        random.shuffle(a) 
        b = [x[0] for x in a]
        c = [x[1] for x in a]
        train_inputAU = torch.stack(b)
        train_inputEMO = torch.stack(c)
        train_rules_input = (train_inputAU, train_inputEMO)
        val_inputAU = torch.cat((data_info['val_input_info']['seen_AU'], data_info['val_input_info']['unseen_AU']))
        val_inputEMO = torch.cat((data_info['val_input_info']['seen_EMO'], data_info['val_input_info']['unseen_EMO']+num_seen))
        val_rules_input = (val_inputAU, val_inputEMO)
        seen_val_inputAU = data_info['val_input_info']['seen_AU']
        seen_val_inputEMO = data_info['val_input_info']['seen_EMO']
        seen_val_rules_input = (seen_val_inputAU, seen_val_inputEMO)
        unseen_val_inputAU = data_info['val_input_info']['unseen_AU']
        unseen_val_inputEMO = data_info['val_input_info']['unseen_EMO']#+num_seen
        unseen_val_rules_input = (unseen_val_inputAU, unseen_val_inputEMO)
        
        train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)
        dataset_EMO = ['happy', 'sad', 'anger', 'surprise', 'fear', 'disgust']
        # dataset_EMO = ['happy', 'sad', 'anger', 'surprise']
        # dataset_EMO = ['fear', 'disgust']
        dataset_info = infolist(dataset_EMO, dataset_AU)

        unseen_priori_rules = get_unseen_priori_rule(train_loader)
        seen_priori_rules = data_info['val_input_info']['seen_priori_rules']
        seen_train_rule_path = os.path.join(pre_seen_rule_path, dataset_name, 'output.pth')
        seen_trained_rule_info = torch.load(seen_train_rule_path, map_location='cpu')
        seen_trained_rules = seen_trained_rule_info['output_rules']
        unseen_train_rule_path = os.path.join(pre_unseen_rule_path, dataset_name, 'output.pth')
        unseen_trained_rule_info = torch.load(unseen_train_rule_path, map_location='cpu')
        unseen_trained_rules = unseen_trained_rule_info['output_rules']
        all_cat_trained_rules = get_cat_rule(seen_trained_rules, unseen_trained_rules)

        output_rules = recouple_rule2(conf, seen_trained_rules, unseen_trained_rules, all_cat_trained_rules)
        
        model = UpdateGraph(conf, all_cat_trained_rules, conf.loc1, conf.loc2, temp=1).to(device)
        val_records = test_rules(conf, model, device, val_rules_input, all_cat_trained_rules, AU_p_d, summary_writer)
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
        val_confu_m2 = val_confu_m.clone()
        for i in range(val_confu_m2.shape[0]):
            val_confu_m2[:, i] = val_confu_m2[:, i] / val_confu_m2[:, i].sum(axis=0)
        infostr_EMO = dataset_info.info_EMO(torch.diag(val_confu_m2).cpu().numpy().tolist())
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
    conf.outdir = os.path.join(conf.outdir, 'seen_trained_cat_unseen_trained', cur_day)

    global device
    conf.gpu = 2
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)
    ensure_dir(conf.outdir)
    set_logger(conf)
    main_CatRule(conf)
    a = 1