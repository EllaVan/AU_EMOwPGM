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

import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='BP4D')
    parser.add_argument('--save_path', type=str, default='save')
    return parser.parse_args()


def run():
    dataset = 'BP4D'
    pkl_basepath = '/media/data1/wf/AU_EMOwPGM/codes/save'
    pkl_path = os.path.join(pkl_basepath, dataset+'/stastics/stat.pkl')
    with open(pkl_path, 'rb') as fo:
        pkl_file = pkl.load(fo)

    EMO = pkl_file['EMO']
    AU = pkl_file['AU']
    interAU = pkl_file['sta_AU_cpt'] 
    prob_AU = pkl_file['sta_prob_AU']
    EMO2AU_cpt = pkl_file['sta_EMO2AU_cpt']
    img_num = pkl_file['img_num']

    EMO2AU_df = pd.DataFrame(EMO2AU_cpt, index=EMO, columns=['AU'+str(i) for i in AU])
    interAU_df = pd.DataFrame(interAU, index=['AU'+str(i) for i in AU], columns=['AU'+str(i) for i in AU])
    probAU_df = pd.DataFrame(prob_AU, index=['AU'+str(i) for i in AU])
    img_num_df = pd.DataFrame(img_num, index=EMO)

    base_path = '/media/data1/wf/AU_EMOwPGM/codes/visulization'
    path = os.path.join(base_path, dataset+'/stat_wR.xlsx')
    file_open= pd.ExcelWriter(path)
    EMO2AU_df.to_excel(file_open, sheet_name='EMO2AU')
    interAU_df.to_excel(file_open, sheet_name='ineterAU')
    probAU_df.to_excel(file_open, sheet_name='prob_AU')
    img_num_df.to_excel(file_open, sheet_name='img_num')
    file_open.save()

    '''
    f, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(EMO2AU_df, annot=True, cmap="YlGnBu", fmt='.3f', ax = ax)
    fig_path = os.path.join(base_path, 'BP4D', 'EMO2AU.jpg')
    plt.savefig(fig_path, dpi=500)
    plt.close()
    '''

    end_flag = True 


if __name__ == '__main__': 
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)

    args = parse_args()
    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print('using gpu:', args.gpu)

    run()
    