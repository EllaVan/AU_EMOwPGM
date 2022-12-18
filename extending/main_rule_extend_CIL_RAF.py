'''
向更多的emotion, 如compound expression等扩展
'''

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
from model_extend.rule_extend2 import learn_rules, test_rules#, generate_seen_sample
# from models.AU_EMO_BP import UpdateGraph_continuous as UpdateGraph
# from models.rule_model import learn_rules, test_rules
from losses import *
from utils import *

def parser2dict():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--fold', type=int, default=0)
    # parser.add_argument('--dataset_order', type=str, default=['RAF-DB', 'BP4D', 'DISFA', 'AffectNet'])
    parser.add_argument('--dataset_order', type=str, default=['RAF-DB-compound'])#, 'DISFA'])
    parser.add_argument('--outdir', type=str, default='save/unseen/CIL_balanced')
    # parser.add_argument('--rule_dir', type=str, default='save/seen')
    parser.add_argument('--rule_dir', type=str, default='../save_balanced/v2/RAF-DB/predsAU_labelsEMO/epoch4_model_fold0.pth')
    # '/media/data1/wf/AU_EMOwPGM/codes/save_balanced/v2/RAF-DB/predsAU_labelsEMO/epoch4_model_fold0.pth'
    parser.add_argument('-b','--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    
    parser.add_argument('-j', '--num_workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--evaluate', action='store_true', help='evaluation mode')

    # --------------------settings for training-------------------
    parser.add_argument('--manualSeed', type=int, default=None)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_idx', type=int, default=20000)
    parser.add_argument('--AUthresh', type=float, default=0.6)
    parser.add_argument('--zeroPad', type=float, default=1e-5)

    parser.add_argument('--pre_train_alpha', type=float, default=0.66)

    parser.add_argument('--isFocal_Loss', type=bool, default=True)
    parser.add_argument('--isClass_Weight', type=bool, default=False)
    parser.add_argument('--isClass_Weight_decay', type=bool, default=False)

    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)

def main(conf):
    num_seen = 4
    num_unseen = 2
    pre_data_path = 'dataset'
    seen_rule_path = 'save/seen/balanced/2022-11-22'
    for dataset_i, dataset_name in enumerate(conf.dataset_order):
        conf.dataset = dataset_name
        conf = get_config(conf)
        cur_outdir = os.path.join(conf.outdir, dataset_name)
        ensure_dir(cur_outdir)
        summary_writer = SummaryWriter(cur_outdir)
        torch.cuda.empty_cache()

        data_path = os.path.join(pre_data_path, dataset_name+'.pkl')
        data_info = torch.load(data_path, map_location='cpu')

        train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
        unseen_priori_rules = get_unseen_priori_rule(train_loader)
        seen_priori_rules = data_info['val_input_info']['seen_priori_rules']
        seen_trained_rule_info = torch.load(os.path.join(seen_rule_path, dataset_name, 'output.pth'), map_location='cpu')
        seen_trained_rules = seen_trained_rule_info['output_rules']
        input_rules = get_complete_rule(seen_trained_rules, unseen_priori_rules)
        
        seen_train_inputAU = data_info['train_input_info']['seen_AU']
        seen_train_inputEMO = data_info['train_input_info']['seen_EMO']
        unseen_train_inputAU = data_info['train_input_info']['unseen_AU']
        unseen_train_inputEMO = data_info['train_input_info']['unseen_EMO']+num_seen
        conf.pre_train_idx = 0
        train_inputAU = torch.cat((seen_train_inputAU, unseen_train_inputAU))
        train_inputEMO = torch.cat((seen_train_inputEMO, unseen_train_inputEMO))
        # train_inputAU = seen_train_inputAU
        # train_inputEMO = seen_train_inputEMO
        # train_inputAU = unseen_train_inputAU
        # train_inputEMO = unseen_train_inputEMO
        
        # samplesAU, samplesEMO = generate_seen_sample(conf, seen_trained_rules[0], seen_trained_rules[-2])
        # repeat_size = unseen_train_inputEMO.shape[0] // len(samplesEMO)
        # samplesAU = samplesAU.repeat(repeat_size, 1)
        # samplesEMO = torch.concat(samplesEMO * repeat_size)

        # conf.pre_train_alpha = 0.66
        # unseen_train_inputAU, unseen_train_inputEMO = shuffle_input(unseen_train_inputAU, unseen_train_inputEMO)
        # pre_train_idx = int(conf.pre_train_alpha*unseen_train_inputEMO.shape[0])
        # conf.pre_train_idx = pre_train_idx
        # part1_unseen_trainAU = unseen_train_inputAU[:pre_train_idx, :]
        # part1_unseen_trainEMO = unseen_train_inputEMO[:pre_train_idx]
        # part2_unseen_trainAU = unseen_train_inputAU[pre_train_idx:, :]
        # part2_unseen_trainEMO = unseen_train_inputEMO[pre_train_idx:]
        # part1_unseen_trainAU, part1_unseen_trainEMO = shuffle_input(part1_unseen_trainAU, part1_unseen_trainEMO)
        # part2_unseen_trainAU = torch.cat((samplesAU, part2_unseen_trainAU))
        # part2_unseen_trainEMO = torch.cat((samplesEMO, part2_unseen_trainEMO))
        # part2_unseen_trainAU, part2_unseen_trainEMO = shuffle_input(part2_unseen_trainAU, part2_unseen_trainEMO)
        # train_inputAU = torch.cat((part1_unseen_trainAU, part2_unseen_trainAU))
        # train_inputEMO = torch.cat((part1_unseen_trainEMO, part2_unseen_trainEMO))

        # samplek = int(unseen_train_inputEMO.shape[0]/num_unseen)
        # samplek = 1000
        # samplesAU, samplesEMO = sample_seen(seen_trained_rules[-2], seen_train_inputAU, seen_train_inputEMO, ori_samplek=samplek)

        # samplesAU = seen_train_inputAU
        # samplesEMO = seen_train_inputEMO

        # train_inputAU = torch.cat((samplesAU, unseen_train_inputAU))
        # train_inputEMO = torch.cat((samplesEMO, unseen_train_inputEMO))

        train_inputAU, train_inputEMO = shuffle_input(train_inputAU, train_inputEMO)

        train_rules_input = (train_inputAU, train_inputEMO)
        val_inputAU = torch.cat((data_info['val_input_info']['seen_AU'], data_info['val_input_info']['unseen_AU']))
        val_inputEMO = torch.cat((data_info['val_input_info']['seen_EMO'], data_info['val_input_info']['unseen_EMO']+num_seen))
        val_rules_input = (val_inputAU, val_inputEMO)
        # seen_val_inputAU = data_info['val_input_info']['seen_AU']
        # seen_val_inputEMO = data_info['val_input_info']['seen_EMO']
        # seen_val_rules_input = (seen_val_inputAU, seen_val_inputEMO)
        # unseen_val_inputAU = data_info['val_input_info']['unseen_AU']
        # unseen_val_inputEMO = data_info['val_input_info']['unseen_EMO']+num_seen
        # unseen_val_rules_input = (unseen_val_inputAU, unseen_val_inputEMO)

        dataset_AU = train_loader.dataset.AU
        priori_AU = train_loader.dataset.priori['AU']
        AU_p_d = (priori_AU, dataset_AU)
        dataset_EMO = ['happy', 'sad', 'anger', 'surprise', 'fear', 'disgust']
        dataset_info = infolist(dataset_EMO, dataset_AU)
        # seen_EMO = ['happy', 'sad', 'anger', 'surprise']
        # seen_info = infolist(seen_EMO, dataset_AU)
        # unseen_EMO = ['fear', 'disgust']
        # unseen_info = infolist(unseen_EMO, dataset_AU)

        conf.lr_relation = 1e-2
        output_rules, train_records, model = learn_rules(conf, train_rules_input, input_rules, seen_trained_rules, AU_p_d, summary_writer)
        # output_rules, train_records, model = learn_rules(conf, train_rules_input, input_rules, AU_p_d, summary_writer)
        train_rules_loss, train_rules_acc, train_confu_m = train_records
        
        # model = UpdateGraph(conf, output_rules, temp=1).to(device)
        # model = UpdateGraph(conf, output_rules).to(device)
        val_records = test_rules(conf, model, val_rules_input, output_rules, AU_p_d, summary_writer)
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

        infostr_rules = {'Dataset: {} val_rules_loss: {:.5f}, val_rules_acc: {:.2f}, val_rules_loss: {:.5f}, val_rules_acc: {:.2f}'
                        .format(dataset_name, train_rules_loss, 100.* train_rules_acc, val_rules_loss, 100.* val_rules_acc)}
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
    conf.outdir = os.path.join(conf.outdir, 'seen_trained_cat_unseen_priori', cur_day)

    global device
    conf.gpu = 1
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)
    ensure_dir(conf.outdir)
    set_logger(conf)
    main(conf)
    a = 1