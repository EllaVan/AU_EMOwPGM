'''
直接把unseen_priori拼接到seen_trained后面,然后两者一起用unseen samples训练
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
# from models.AU_EMO_BP import UpdateGraph_continuous as UpdateGraph
from model_extend.rule_extend import UpdateGraph_proj as UpdateGraph
# from models.rule_model import learn_rules, test_rules
from model_extend.rule_extend import learn_rules_KL as learn_rules
from model_extend.rule_extend import proj_func, test_rules_dis, test_rules
from losses import *
from utils import *
from utils_extend import *

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

def main(conf):
    num_seen = 4
    num_unseen = 2
    pre_data_path = 'dataset'
    seen_rule_path = 'save/seen/2022-11-08'
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        conf.dataset = dataset_name
        conf = get_config(conf)
        cur_outdir = os.path.join(conf.outdir, dataset_name)
        ensure_dir(cur_outdir)
        summary_writer = SummaryWriter(cur_outdir)
        torch.cuda.empty_cache()

        data_path = os.path.join(pre_data_path, dataset_name+'.pkl')
        data_info = torch.load(data_path, map_location='cpu')#['state_dict']
        train_inputAU = data_info['train_input_info']['unseen_AU']
        train_inputEMO = data_info['train_input_info']['unseen_EMO']#+num_seen
        a = list(zip(train_inputAU, train_inputEMO))
        random.shuffle(a) 
        b = [x[0] for x in a]
        c = [x[1] for x in a]
        train_inputAU = torch.stack(b)
        train_inputEMO = torch.stack(c)
        train_rules_input = (train_inputAU, train_inputEMO)
        # val_inputAU = torch.cat((data_info['val_input_info']['seen_AU'], data_info['val_input_info']['unseen_AU']))
        # val_inputEMO = torch.cat((data_info['val_input_info']['seen_EMO'], data_info['val_input_info']['unseen_EMO']+num_seen))
        val_inputAU = data_info['val_input_info']['unseen_AU']
        val_inputEMO = data_info['val_input_info']['unseen_EMO']
        val_rules_input = (val_inputAU, val_inputEMO)
        
        train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
        unseen_priori_rules = get_unseen_priori_rule(train_loader)
        seen_priori_rules = data_info['val_input_info']['seen_priori_rules']
        seen_trained_rule_info = torch.load(os.path.join(seen_rule_path, dataset_name, 'output.pth'), map_location='cpu')
        seen_trained_rules = seen_trained_rule_info['output_rules']
        input_rules = get_complete_rule(seen_trained_rules, unseen_priori_rules)

        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)

        conf.lr_relation = 1e-4
        output_rules, train_records, model = learn_rules(conf, device, train_rules_input, unseen_priori_rules, seen_trained_rules, AU_p_d, summary_writer)
        train_rules_loss, train_rules_acc, train_confu_m = train_records

        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)
        # dataset_EMO = ['happy', 'sad', 'anger', 'surprise', 'fear', 'disgust']
        # dataset_EMO = ['happy', 'sad', 'anger', 'surprise']
        dataset_EMO = ['fear', 'disgust']
        dataset_info = infolist(dataset_EMO, dataset_AU)
        
        model = UpdateGraph(conf, output_rules, conf.loc1, conf.loc2, temp=1).to(device)
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
    conf.outdir = os.path.join(conf.outdir, 'only_unseen', cur_day)

    global device
    conf.gpu = 0
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)
    ensure_dir(conf.outdir)
    set_logger(conf)
    main(conf)
    a = 1