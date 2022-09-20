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
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
from materials.process_priori import cal_interAUPriori


class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        
        weights = dataset.train_weighs

        self.weights = torch.DoubleTensor(list(weights))

    def _get_labels(self, dataset):
        return [dataset.label[i] for i in self.indices]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class BP4DDataSet(data.Dataset):
    def __init__(self, phase=None, label_index=None):
        self.phase = phase
        self.pkl_path = '/media/data1/wf/AU_EMOwPGM/codes/dataset/BP4D/rules_learn_pkl/BP4D.pkl'
        with open(self.pkl_path, 'rb') as fo:
            pkl_file = pkl.load(fo)
        self.EMOCode = pkl_file['EMOCode']
        self.priori = pkl_file['priori']
        self.train_weighs = pkl_file['train_weights']

        if self.phase == 'train':
            self.AUlists = pkl_file['train']
        elif self.phase== 'test':
            self.AUlists = pkl_file['test'][label_index]
        elif self.phase== 'all_test':
            self.AUlists = pkl_file['all_test'][label_index]

    def __len__(self):
        return len(self.AUlists)

    def __getitem__(self, idx):
        info = self.AUlists[idx]
        emo_label = info[0]
        au_evidence = info[1]
        return au_evidence, emo_label, idx
    
        
def getBP4Ddata(all_test=False):
    train_dataset = BP4DDataSet(phase='train')
    num_EMO = len(train_dataset.priori['EMO'])
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, pin_memory=True)
    
    if all_test is False:
        test_phase = 'test'
    else:
        test_phase = 'all_test'
    test_dataset = []
    test_loader = []
    test_len = 0
    for k in range(num_EMO):
        test_dataset.append(BP4DDataSet(phase=test_phase, label_index=k))
        test_loader.append(torch.utils.data.DataLoader(test_dataset[k], shuffle=False, pin_memory=True))
        test_len += test_dataset[k].__len__()
    print('Test set size:', test_len)

    return train_loader, test_loader


class CASMEDataSet(data.Dataset):
    def __init__(self, phase=None, label_index=None):
        self.phase = phase
        self.pkl_path = '/media/data1/wf/AU_EMOwPGM/codes/dataset/CASME/CAMSE_nR.pkl'
        with open(self.pkl_path, 'rb') as fo:
            pkl_file = pkl.load(fo)
        self.priori = pkl_file['priori']
        self.EMO = pkl_file['EMO']
        if self.phase == 'train':
            self.AUlists = pkl_file['train']
        elif self.phase== 'test':
            self.AUlists = pkl_file['test'][label_index]
        elif self.phase== 'all_test':
            self.AUlists = pkl_file['all_test'][label_index]

    def __len__(self):
        return len(self.AUlists)

    def __getitem__(self, idx):
        info = self.AUlists[idx]
        emo_label = info[0]
        au_evidence = np.array(info[1])
        return au_evidence, emo_label, idx
        
def getCASMEdata(num_EMO=6, all_test=False):
    train_dataset = CASMEDataSet(phase='train')
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, pin_memory=True)

    if all_test is False:
        test_phase = 'test'
    else:
        test_phase = 'all_test'
    num_EMO = len(train_dataset.EMO)
    test_dataset = []
    test_loader = []
    test_len = 0
    for k in range(num_EMO):
        test_dataset.append(CASMEDataSet(phase=test_phase, label_index=k))
        test_loader.append(torch.utils.data.DataLoader(test_dataset[k], shuffle=False, pin_memory=True))
        test_len += test_dataset[k].__len__()
    print('Test set size:', test_len)
    return train_loader, test_loader


if __name__ == '__main__':
    # train_loader, test_loader = getBP4Ddata()
    train_loader, test_loader = getCASMEdata()
    end_flag = True
        