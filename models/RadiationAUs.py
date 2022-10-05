import os
import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import numpy as np
import pickle as pkl
import random

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
import networkx as nx

from materials.process_priori import cal_interAUPriori

import torch
import torch.nn as nn
import torch.nn.functional as F

def RadiateAUs(AU_cpt, occAU, prob_occAU=None, thresh=0.6, num_EMO=6):
    if prob_occAU is None:
        prob_occAU = [[1]] * len(occAU)
    ra_au_1 = AU_cpt[:, occAU].reshape(len(occAU), -1) * prob_occAU
    ra_au_1 = np.where(ra_au_1 > thresh, ra_au_1, 0)

    prob_occAU1 = np.mean(ra_au_1, axis=0).reshape(-1, 1)
    prob_all_au = prob_occAU1.copy()
    prob_all_au[occAU, :] = prob_occAU
    
    return prob_all_au

def static_op(conf, emo_label, prob_all_au, loc2, EMO2AU):
    for i in loc2:
        if EMO2AU[emo_label, i] - conf.zeroPad > 0:
            prob_all_au[i, :] = 1
    # if dataset == 'BP4D':
    #     if emo_label == 0 or emo_label == 2 or emo_label == 4:
    #         prob_all_au[-2, :] = 1
    #     if emo_label == 2 or emo_label == 4:
    #         prob_all_au[-1, :] = 1

    return prob_all_au

def RadiateAUs_v2(conf, emo_label, AU_cpt, occAU, loc2, EMO2AU, prob_occAU=None, thresh=0.6, num_EMO=6):
    if prob_occAU is None:
        prob_occAU = [[1]] * len(occAU)
    ra_au_1 = AU_cpt[:, occAU].reshape(len(occAU), -1) * prob_occAU
    ra_au_1 = np.where(ra_au_1 > thresh, ra_au_1, 0)

    prob_occAU1 = np.mean(ra_au_1, axis=0).reshape(-1, 1)
    prob_all_au = prob_occAU1.copy()
    prob_all_au[occAU, :] = prob_occAU
    prob_all_au = static_op(conf, emo_label, prob_all_au, loc2, EMO2AU)
    
    return prob_all_au


def do_RadiateAUs(AU_cpt, occ_au, thresh):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh)

class interAUs(BayesianNetwork):

    def new_edge(self, u, v):
        if nx.has_path(self, v, u):
            pass
        else:
            self.add_edge(u, v)

    def __init__(self, input_AU_cpt, AU, thresh=0.6):
        super(interAUs, self).__init__()
        self.AU = AU
        self.thresh = thresh
        self.input_AU_cpt = np.where(input_AU_cpt > self.thresh, input_AU_cpt, 0)
        mask = np.nonzero(self.input_AU_cpt)
        self.pos = tuple(zip(mask[0], mask[1]))

        # self.interAU_model = BayesianNetwork()
        # 建立节点
        nodes = ['AU'+str(au) for au in AU]
        self.add_nodes_from(nodes)

        # 建立边
        factor_len = len(AU)
        input = self.input_AU_cpt.reshape(1, -1).squeeze(0)
        sort_list = np.argsort(-input)
        for i in range(len(self.pos)):
            head = sort_list[i]%factor_len
            tail = sort_list[i]//factor_len
            edge = ('AU'+str(AU[head]), 'AU'+str(AU[tail]))
            self.new_edge(edge[0], edge[1])
            
        print('The interAUs model is established, edges are ', self.edges)
        pass


def do_interAUs():
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    nodes = ['AU'+str(au) for au in AU]
    
    base_path = '/media/data1/wf/AU_EMOwPGM/codes/save'
    task_path = 'newEMO2AUwP(AU)_staticAU256_occAU!=0_LR=0.001+0.0001at20000_AllMultiFactor_interAUwPGM'
    base_path = os.path.join(base_path, task_path)
    pkl_path = os.path.join(base_path, 'results.pkl')
    with open(pkl_path, 'rb') as fo:
        pkl_file = pkl.load(fo)
    interAU_model = pkl_file['interAU_model']
    # interAU_model = interAUs(AU_cpt, AU)

    evidence={'AU1':1, 'AU2':1}
    query_node = nodes.copy()
    query_node.remove('AU1')
    query_node.remove('AU2')
    interAU_infer = VariableElimination(interAU_model)
    q = interAU_infer.query(variables=query_node, evidence=evidence, joint=False, show_progress=False)


class interAUGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, w=None, thresh=0.6):
        super().__init__()
        self.outdimension = out_channels
        self.thresh = thresh

        if w is None:
            self.w = nn.Parameter(torch.empty(in_channels, out_channels))
            torch.nn.init.kaiming_normal_(self.w, a=0, mode='fan_in', nonlinearity='relu')
        else:
            self.w = nn.Parameter(w)

    def forward(self, inputs, adj, prob_AU):
        outputs = torch.mm(torch.mm(adj, self.w/17.0), inputs)
        m = activate()
        outputs, pos = m(outputs)

        for i in range(outputs.shape[0]):
            outputs[i] = outputs[i] / prob_AU[i]
        return outputs, pos

class interAURelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        pos = torch.where(inp > 0.6)[0]
        inp = torch.where(inp <= 0.6, torch.ones_like(inp), inp)
        inp = torch.where(inp > 1, torch.ones_like(inp), inp)
        return inp, pos

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        return grad_output * torch.where(inp <= 0.6, torch.zeros_like(inp),
                                         torch.ones_like(inp))

class activate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out, pos = interAURelu.apply(x)
        return out, pos


def do_interAUGraphConv():
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    input_occAU = np.zeros((len(AU), 1))
    input_occAU[:2] = 1
    conv = interAUGraphConv(len(AU), len(AU))

    input_occAU = torch.tensor(input_occAU, dtype=torch.float32)
    AU_cpt = torch.tensor(AU_cpt, dtype=torch.float32)

    outputs = conv(input_occAU, AU_cpt)
    end_flag = True


if __name__ == '__main__': 
    do_interAUGraphConv()
    end_flag = True
