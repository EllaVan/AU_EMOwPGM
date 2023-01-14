import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
need_path = [current_dir, parent_dir, os.path.join(parent_dir,'models'), os.path.join(parent_dir,'extending')]
sys.path = need_path + sys.path
os.chdir(current_dir)

from models.AU_EMO_BP import UpdateGraph_continuous as UpdateGraph
from models.rule_model import learn_rules, test_rules

from conf import ensure_dir
from utils import getDatasetInfo
from model_extend.utils_extend import get_config
from main_rule_extend_CIL import parser2dict

import torch
from tensorboardX import SummaryWriter

import numpy as np

def differ_seen_all(dataset_name='BP4D'):
    pre_all_trained_name = 'labelsAU_labelsEMO/epoch4_model_fold0.pth'

    pre_seen_trained_path = 'save/seen/2022-11-08_v2'
    seen_trained_path = os.path.join(pre_seen_trained_path, dataset_name, 'output.pth')
    seen_trained_info = torch.load(seen_trained_path, map_location='cpu')

    pre_all_trained_path1 = '../results'
    pre_all_trained_path2 = 'Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    all_trained_path = os.path.join(pre_all_trained_path1, dataset_name, pre_all_trained_path2, pre_all_trained_name)
    all_trained_info = torch.load(all_trained_path, map_location='cpu')

    pre_cat_all_trained_path = 'save/unseen/CIL/2022-11-10/seen_trained_cat_unseen_priori'
    cat_all_trained_path = os.path.join(pre_cat_all_trained_path, dataset_name, 'output.pth')
    cat_all_trained_info = torch.load(cat_all_trained_path, map_location='cpu')

    seen_trained_rule = seen_trained_info['output_rules']
    all_trained_rule = all_trained_info['output_rules']
    cat_all_trained_rule = cat_all_trained_info['output_rules']

    return seen_trained_rule, all_trained_rule, cat_all_trained_rule

def test_seen():
    conf = parser2dict()
    global device
    conf.gpu = 3
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)

    dataset_name = 'BP4D'
    conf.dataset = dataset_name
    conf = get_config(conf)

    data_path = os.path.join('dataset', dataset_name+'.pkl')
    data_info = torch.load(data_path, map_location='cpu')
    val_inputAU = data_info['val_input_info']['seen_AU']
    val_inputEMO = data_info['val_input_info']['seen_EMO']
    val_rules_input = (val_inputAU, val_inputEMO)

    train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
    dataset_AU = train_loader.dataset.AU
    priori_AU = train_loader.dataset.priori['AU']
    AU_p_d = (priori_AU, dataset_AU)
    
    pre_outdir = 'tmp_test/seen'
    seen_trained_rule, all_trained_rule, cat_all_trained_rule = differ_seen_all(dataset_name)
    # cur_outdir = os.path.join(pre_outdir, 'seen_trained')
    # summary_writer = SummaryWriter(cur_outdir)
    # model_seen = UpdateGraph(conf, seen_trained_rule, conf.loc1, conf.loc2).to(device)
    # seen_trained_val_records = test_rules(conf, model_seen, device, val_rules_input, seen_trained_rule, AU_p_d, summary_writer)
    # seen_trained_val_rules_loss, seen_trained_val_rules_acc, seen_trained_val_confu_m = seen_trained_val_records
    # cur_outdir = os.path.join(pre_outdir, 'all_trained')
    # summary_writer = SummaryWriter(cur_outdir)
    # model_all = UpdateGraph(conf, all_trained_rule, conf.loc1, conf.loc2).to(device)
    # all_trained_val_records = test_rules(conf, model_all, device, val_rules_input, all_trained_rule, AU_p_d, summary_writer)
    # all_trained_val_rules_loss, all_trained_val_rules_acc, all_trained_val_confu_m = all_trained_val_records
    cur_outdir = os.path.join(pre_outdir, 'cat_all_trained')
    summary_writer = SummaryWriter(cur_outdir)
    model_cat_all = UpdateGraph(conf, cat_all_trained_rule, conf.loc1, conf.loc2).to(device)
    cat_all_trained_val_records = test_rules(conf, model_cat_all, device, val_rules_input, cat_all_trained_rule, AU_p_d, summary_writer)
    cat_all_trained_val_rules_loss, cat_all_trained_val_rules_acc, cat_all_trained_val_confu_m = cat_all_trained_val_records

    # seen_trained_val_confu_m_2 = seen_trained_val_confu_m.clone()
    # for i in range(seen_trained_val_confu_m_2.shape[0]):
    #     seen_trained_val_confu_m_2[:, i] = seen_trained_val_confu_m_2[:, i] / seen_trained_val_confu_m_2[:, i].sum(axis=0)
    # all_trained_val_confu_m_2 = all_trained_val_confu_m.clone()
    # for i in range(all_trained_val_confu_m_2.shape[0]):
    #     all_trained_val_confu_m_2[:, i] = all_trained_val_confu_m_2[:, i] / all_trained_val_confu_m_2[:, i].sum(axis=0)
    cat_all_trained_val_confu_m_2 = cat_all_trained_val_confu_m.clone()
    for i in range(cat_all_trained_val_confu_m_2.shape[0]):
        cat_all_trained_val_confu_m_2[:, i] = cat_all_trained_val_confu_m_2[:, i] / cat_all_trained_val_confu_m_2[:, i].sum(axis=0)

    a = 1

def view_file():
    path = 'save/unseen/CIL/2022-11-09/v1_12/BP4D/output.pth'
    file_info = torch.load(path, map_location='cpu')
    a = 1

def chuli_basic():
    basic_rule_path = '/media/data1/wf/AU_EMOwPGM/codes/save_balanced/v2/RAF-DB/predsAU_labelsEMO/epoch4_model_fold0.pth'
    file_info_basic = torch.load(basic_rule_path, map_location='cpu')
    seen_path = 'save/seen/balanced/2022-12-11_v2/RAF-DB/output.pth'
    file_info_seen = torch.load(seen_path, map_location='cpu')
    end_flag = 1


def temp1():
    pre_data_path = 'dataset'
    dataset_name = 'BP4D_all'
    data_path = os.path.join(pre_data_path, dataset_name+'.pkl')
    data_info = torch.load(data_path, map_location='cpu')
    seen_AU = data_info['train_input_info']['seen_AU']
    seen_EMO = data_info['train_input_info']['seen_EMO']
    unseen_AU = data_info['train_input_info']['unseen_AU']
    unseen_EMO = data_info['train_input_info']['unseen_EMO']
    record = []
    for i in range(6):
        loc = torch.where(seen_EMO==i)[0]
        AU_i = seen_AU[loc, :]
        stastic_AUi = torch.sum(AU_i, dim=0)/AU_i.shape[0]
        record_AUi = stastic_AUi.reshape(1, -1)
        record.append(record_AUi)
    for i in range(2):
        loc = torch.where(unseen_EMO==i+6)[0]
        AU_i = seen_AU[loc, :]
        stastic_AUi = torch.sum(AU_i, dim=0)/AU_i.shape[0]
        record_AUi = stastic_AUi.reshape(1, -1)
        record.append(record_AUi)
    record = np.concatenate(record)
    end_flag = 1

def read1():
    path = ''


if __name__=='__main__':
    # differ_seen_all()
    # test_seen()
    # view_file()
    # chuli_basic()
    temp1()
    a = 1
