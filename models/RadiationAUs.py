import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('/media/database/data4/wf/AU_EMOwPGM/codes')
import numpy as np

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation

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


def BuildAUs(AU_cpt, occAU, prob_occAU=None, thresh=0.6):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    AU = list(map(int, AU))
    interAU_edges = []
    num_edges = 0
    interAU_prob = {}
    for i in range(AU_cpt.shape[0]):
        for j in range(AU_cpt.shape[1]):
            if AU_cpt[i][j] > thresh:
                num_edges += 1
                interAU_edges.append(('AU'+str(AU[j]), 'AU'+str(AU[i])))
                interAU_prob[num_edges] = AU_cpt[i][j]
    interAU_model = BayesianNetwork(interAU_edges)
    


    if prob_occAU is None:
        prob_occAU = [[1]] * len(occAU)
    ra_au_1 = AU_cpt[occAU, :].reshape(len(occAU), -1) * prob_occAU
    prob_occAU1 = np.mean(ra_au_1, axis=0)
    # prob_occAU1 = np.max(ra_au_1, axis=0).reshape(-1, 1)
    prob_all_au = prob_occAU1.copy()
    occAU_2 = set(np.where(prob_occAU1>=thresh)[0])-set(occAU)
    if len(occAU_2) != 0:
        ra_au_2 = AU_cpt[list(occAU_2), :].reshape(len(occAU_2), -1) * prob_occAU1
        prob_occAU2 = np.mean(ra_au_2, axis=0)
        # prob_occAU2 = np.max(ra_au_2, axis=0).reshape(-1, 1)
        prob_all_au = np.max(((prob_occAU1+prob_occAU2) / 2.0), axis=0).reshape(-1, 1)
        prob_all_au[occAU, :] = prob_occAU
        prob_all_au[list(occAU_2), :] = prob_occAU1[list(occAU_2)].reshape(-1, 1)
    else:
        prob_all_au[occAU, :] = prob_occAU
    return prob_all_au


if __name__ == '__main__':
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    RadiateAUs(AU_cpt, [1, 2, 3])
