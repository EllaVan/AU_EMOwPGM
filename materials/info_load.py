import os
import argparse
import pickle as pkl
import random
from sklearn.utils import shuffle

import torch
import torch.utils.data as data

import numpy as np
import pandas as pd

import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('/media/database/data4/wf/AU_EMOwPGM/codes')
from materials.process_priori import cal_interAUPriori

class BP4DDataSet(data.Dataset):
    def __init__(self, phase=None, label_index=None):
        self.phase = phase
        self.pkl_path = '/media/database/data4/wf/AU_EMOwPGM/codes/dataset/BP4D.pkl'
        with open(self.pkl_path, 'rb') as fo:
            pkl_file = pkl.load(fo)
        self.EMOCode = pkl_file['EMOCode']
        self.priori = pkl_file['priori']

        if self.phase == 'train':
            self.AUlists = pkl_file['train']
        elif self.phase== 'test':
            self.AUlists = pkl_file['test'][label_index]

    def __len__(self):
        return len(self.AUlists)

    def __getitem__(self, idx):
        info = self.AUlists[idx]
        emo_label = info[0]
        au_evidence = info[1]
        return au_evidence, emo_label, idx
        
def getBP4Ddata(num_EMO=6):
    train_dataset = BP4DDataSet(phase='train')
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, pin_memory=True)

    test_dataset = []
    test_loader = []
    test_len = 0
    for k in range(num_EMO):
        test_dataset.append(BP4DDataSet(phase='test', label_index=k))
        test_loader.append(torch.utils.data.DataLoader(test_dataset[k], shuffle=False, pin_memory=True))
        test_len += test_dataset[k].__len__()
    print('Test set size:', test_len)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = getBP4Ddata()
    end_flag = True
        