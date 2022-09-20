import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

import argparse
import datetime
from errno import EMULTIHOP
import pytz
import os
import shutil
import pickle as pkl
import random

import csv
import pandas as pd
import numpy as np

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    pre_path = '/media/data1/wf/AU_EMOwPGM/codes/results/AffectNet/Test/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    info_type_path = 'predsAU_predsEMO'
    info_source_path = 'epoch20_model_fold0'
    rules_path = os.path.join(pre_path, info_type_path, info_source_path+'.pth')
    rules = torch.load(rules_path, map_location='cpu')['output_rules']
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = rules

    EMO2AU_df = pd.DataFrame(EMO2AU_cpt, index=EMO, columns=['AU'+str(i) for i in AU])
    interAU_df = pd.DataFrame(AU_cpt, index=['AU'+str(i) for i in AU], columns=['AU'+str(i) for i in AU])
    probAU_df = pd.DataFrame(prob_AU, index=['AU'+str(i) for i in AU])

    '''
    file_open= pd.ExcelWriter(path)
    EMO2AU_df.to_excel(file_open, sheet_name='EMO2AU')
    interAU_df.to_excel(file_open, sheet_name='ineterAU')
    probAU_df.to_excel(file_open, sheet_name='prob_AU')
    file_open.save()
    '''
    
    f, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(EMO2AU_df, annot=True, cmap="YlGnBu", fmt='.3f', ax = ax)
    fig_path = os.path.join(os.path.join(pre_path, info_type_path, info_source_path), 'EMO2AU.jpg')
    plt.savefig(fig_path, dpi=500)
    plt.close()
    

    end_flag = True 


if __name__ == '__main__': 
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)

    run()