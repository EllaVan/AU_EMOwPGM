import os
import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('/media/database/data4/wf/AU_EMOwPGM/codes')
import numpy as np
import pickle as pkl

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
import networkx as nx

from materials.process_priori import cal_interAUPriori

def RadiateAUs(AU_cpt, occAU, prob_occAU=None, thresh=0.6):
    if prob_occAU is None:
        prob_occAU = [[1]] * len(occAU)
    ra_au_1 = AU_cpt[:, occAU].reshape(len(occAU), -1) * prob_occAU
    # prob_occAU1 = np.mean(ra_au_1, axis=0)
    prob_occAU1 = np.max(ra_au_1, axis=0).reshape(-1, 1)
    prob_all_au = prob_occAU1.copy()
    prob_all_au[occAU, :] = prob_occAU
    # occAU_2 = set(np.where(prob_occAU1>=thresh)[0])-set(occAU)
    # if len(occAU_2) != 0:
    #     ra_au_2 = AU_cpt[list(occAU_2), :].reshape(len(occAU_2), -1) * prob_occAU1
    #     prob_occAU2 = np.mean(ra_au_2, axis=0)
    #     # prob_occAU2 = np.max(ra_au_2, axis=0).reshape(-1, 1)
    #     prob_all_au = np.max(((prob_occAU1+prob_occAU2) / 2.0), axis=0).reshape(-1, 1)
    #     prob_all_au[occAU, :] = prob_occAU
    #     prob_all_au[list(occAU_2), :] = prob_occAU1[list(occAU_2)].reshape(-1, 1)
    # else:
    #     prob_all_au[occAU, :] = prob_occAU
    return prob_all_au

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
    

if __name__ == '__main__':
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    nodes = ['AU'+str(au) for au in AU]
    
    base_path = '/media/database/data4/wf/AU_EMOwPGM/codes/save'
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
    
    end_flag = True
