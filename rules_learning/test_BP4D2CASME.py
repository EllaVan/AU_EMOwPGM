import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

import torch

from models.RadiationAUs import RadiateAUs, interAUs
from rules_learning.dataset_process import info_load
from models.AU_EMO_BP import UpdateGraph
from materials.process_priori import cal_interAUPriori
import rules_learning.utils as utils


def infer_trained():
    pkl_path = '/media/data1/wf/AU_EMOwPGM/codes/save/CASME/stastics/stat_nR.pkl'
    with open(pkl_path, 'rb') as fo:
        data_pkl = pkl.load(fo)
    EMO = data_pkl['EMO']
    AU = data_pkl['AU']
    
    '''
    prob_AU = data_pkl['sta_prob_AU']
    EMO2AU_cpt = data_pkl['sta_EMO2AU_cpt']
    AU_cpt = data_pkl['sta_AU_cpt']
    '''

    
    trained_path = '/media/data1/wf/AU_EMOwPGM/codes/save/BP4D/2022-07-30/results.pkl'
    with open(trained_path, 'rb') as fo:
        trained_pkl = pkl.load(fo)
    prob_AU = trained_pkl['new_prob_AU']
    EMO2AU_cpt = trained_pkl['new_EMO2AU']
    AU_cpt = trained_pkl['new_AU_cpt']
    

    '''
    BP4Dstat_path = '/media/data1/wf/AU_EMOwPGM/codes/save/BP4D/stastics/stat_basicEMO.pkl'
    with open(BP4Dstat_path, 'rb') as fo:
        stat_file = pkl.load(fo)
    a = stat_file['sta_EMO2AU_cpt']
    delete_index = [3]
    a = np.delete(a, delete_index, axis=0)
    a = a[[0, 1, 3, 4, 2, 5], :]
    EMO2AU_cpt[:, :-2] = a
    AU_cpt[:-2, :-2] = stat_file['sta_AU_cpt']
    prob_AU[:-2] = stat_file['sta_prob_AU']
    '''
    
    '''
    EMO2AU_cpt, prob_AU, _, AU_cpt, _, _ = cal_interAUPriori()
    delete_index = [3]
    EMO2AU_cpt = np.delete(EMO2AU_cpt, delete_index, axis=0)
    for i, j in enumerate(AU[:-2]):
        prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
    prob_AU = np.where(prob_AU > 0, prob_AU, 1e-4)
    prob_AU = np.where(prob_AU <= 1, prob_AU, 1)
    '''

    confu_m = torch.zeros((6, 6)) # confu_m = torch.zeros((len(EMO), len(EMO)))
    _, test_loader = info_load.getCASMEdata(all_test=True)
    acc_all = []
    for k, emo in enumerate(EMO):
        acc_record = []
        for idx, (cur_item, emo_label, index) in enumerate(test_loader[k], 1):
            if k == 3 or k== 4:
                emo_label += 1
            weight = []
            occ_au = []
            prob_all_au = np.zeros((len(AU),))
            if len(cur_item.shape) > 2:
                cur_item = cur_item.squeeze(0)
            for i, au in enumerate(AU):
                if cur_item[0, au] == 1:
                    occ_au.append(i)
                    prob_all_au[i] = 1
                weight.append(EMO2AU_cpt[:, i])

            if len(occ_au) != 0:
                # prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
                
                pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
                weight = np.array(weight)
                
                for i in range(prob_all_au.shape[0]-2):
                    if i in pos:
                        prob_all_au[i] = prob_all_au[i] / prob_AU[i]
                    else:
                        prob_all_au[i] = 1 / (1-prob_AU[i])
                        weight[i, :] = 1 - weight[i, :]

                prob_all_au[-2] = 1 / prob_AU[-2]
                prob_all_au[-1] = 1 / prob_AU[-1]
                if emo_label == 0:
                    weight[-1, :] = 1 - weight[-1, :]
                    prob_all_au[-1] = 1 / (1-prob_AU[-1])
                elif emo_label == 2 or emo_label == 4:
                    pass
                else:
                    weight[-2, :] = 1 - weight[-2, :]
                    weight[-1, :] = 1 - weight[-1, :]
                    prob_all_au[-2] = 1 / (1-prob_AU[-2])
                    prob_all_au[-1] = 1 / (1-prob_AU[-1])
                init = torch.ones((1, weight.shape[1])) # init = torch.ones((1, len(EMO)))
                for i in range(weight.shape[1]):
                    for j in range(1, 3):
                        init[:, i] *= weight[-j][i]*prob_all_au[-j]
                
                weight = np.where(weight > 0, weight, 1e-4)
                
                update = UpdateGraph(device, in_channels=1, out_channels=6, W=weight, 
                                    prob_all_au=prob_all_au, init=init).to(device)
                
                AU_evidence = torch.ones((1, 1)).to(device)
                cur_prob = update(AU_evidence)
                cur_pred = torch.argmax(cur_prob)
                # confu_m = utils.confusion_matrix(cur_pred, labels=emo_label, conf_matrix=confu_m)
                confu_m[cur_pred, emo_label] += 1
                cur_p = torch.eq(cur_pred, emo_label.to(device)).type(torch.FloatTensor).item()
                acc_record.append(cur_p)
        print('The Acc of %s is %.5f' %(emo, np.array(acc_record).mean()))
        acc_all += acc_record
    print('The Acc of dataset is %.5f' %(np.array(acc_all).mean()))
    # utils.plot_confusion_matrix(confu_m.numpy(), classes=EMO, normalize=True,
    #                              title='Normalized confusion matrix')

    end_flag = 0


if __name__ == '__main__':
    global device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    infer_trained()
    end_flag = True