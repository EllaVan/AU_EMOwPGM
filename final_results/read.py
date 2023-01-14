import os
import sys
current_dir = os.path.dirname(__file__) # 当前文件所属文件夹
os.chdir(current_dir)

import argparse
from easydict import EasyDict as edict
import yaml

import numpy as np
import torch

def parser2dict():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--dataset_order', type=str, default=['RAF-DB-compound'])
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)

# 根据不同的数据集取不同的数据
def getdata(dataset_name):
    if dataset_name == 'BP4D':
        cur_path = 'epoch4_model_fold0.pth'
        info_source = ['labelsAU_record', 'labelsEMO_record']
    elif dataset_name == 'RAF-DB':
        cur_path = 'epoch4_model_fold0.pth'
        info_source = ['predsAU_record', 'labelsEMO_record']
    elif dataset_name == 'AffectNet':
        cur_path = 'epoch1_model_fold0.pth'
        info_source = ['predsAU_record', 'labelsEMO_record']
    elif dataset_name == 'DISFA':
        cur_path = 'epoch4_model_fold0.pth'
        info_source = ['labelsAU_record', 'predsEMO_record']
    return cur_path, info_source

def main():
    pre_path1 = '/media/data1/wf/AU_EMOwPGM/codes/final_results/v2'
    # pre_path2 = 'v2/BP4D/labelsAU_labelsEMO/epoch4_model_fold0.pth'
    datasets = ['BP4D', 'RAF-DB', 'AffectNet', 'DISFA']
    store_list = [] * 6

    for dataset_name in datasets:
        print(dataset_name)
        checkpoint = {}
        epoch_path, info_source = getdata(dataset_name)
        info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
        info_path = os.path.join(pre_path1, dataset_name, info_source_path, epoch_path)
        all_info = torch.load(info_path, map_location='cpu')#['state_dict']

        output_rules = all_info['output_rules']
        EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = output_rules
        assert EMO2AU_cpt.shape[0] == len(EMO)
        for emo_i in range(EMO2AU_cpt.shape[0]):
            sort_EMO2AU = EMO2AU_cpt[emo_i, :].copy()
            cur_EMO2AU = list(np.argsort(-sort_EMO2AU))
            print(EMO[emo_i], end=': ')
            for k, au_j in enumerate(cur_EMO2AU):
                if (k+1) % 6 == 0:
                    end_note = '| '
                else:
                    end_note = ', '
                print('AU' + str(AU[au_j]), end=end_note)
            print()
        a = 1
    a = 1
    
if __name__=='__main__':
    main()