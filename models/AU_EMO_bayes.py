import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from InferPGM.inference import VariableElimination

import numpy as np

from materials.process_priori import cal_interAUPriori


class AU_EMO_bayesGraph():
    def __init__(self, EMO2AU_cpt, prob_AU, EMO, AU):
        self.AU = list(map(int, AU))
        self.EMO = EMO
        self.prob_AU = prob_AU
        self.EMO2AU_cpt = EMO2AU_cpt

        self.AU_EMO_jpt = EMO2AU_cpt*1.0/len(EMO)
        self.AU2EMO_cpt = []
        for i in range(len(AU)):
            self.AU2EMO_cpt.append(list(self.AU_EMO_jpt[:, i]/prob_AU[i]))
        
        # 建立由EMO指向AU的概率图
        EMO2_AU = []
        for i in AU:
            EMO2_AU.append(('EMO', 'AU'+str(i)))
        self.AU_EMO_model = BayesianNetwork(EMO2_AU)
        # EMO的条件概率，由于EMO没有父亲节点，故值为EMO的边缘概率
        EMO_cpd = TabularCPD(variable='EMO', variable_card=len(EMO), values=[[1.0/len(EMO)]]*len(EMO))
        self.AU_EMO_model.add_cpds(EMO_cpd)
        # 对于每一个AU
        AU_cpd = {}
        for i, j in enumerate(AU):
            var = 'AU'+str(j)
            AU_cpd[j] = TabularCPD(variable=var, variable_card=2, 
                                values=[[1-value for value in list(EMO2AU_cpt[:, i])], list(EMO2AU_cpt[:, i])],
                                evidence=['EMO'], evidence_card=[len(EMO)]
                                ) # P(EMO | AU)
            self.AU_EMO_model.add_cpds(AU_cpd[j])

        if self.AU_EMO_model.check_model(): #确认模型是否正确
            print('The established AU-EMO PGM is Valid') 
        else:
            raise NotImplementedError('The established AU-EMO PGM is NOT valid')
        
    def infer(self, AU_evidence):
        self.model_infer = VariableElimination(self.AU_EMO_model)
        self.out_prob = self.model_infer.query(variables=['EMO'], evidence=AU_evidence)
        return self.out_prob


def initGraph():
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()

    c = list(map(int, AU))
    AU = c

    AU_EMO_jpt = EMO2AU_cpt*1.0/len(EMO)
    AU2EMO_cpt = []
    for i in range(len(AU)):
        AU2EMO_cpt.append(list(AU_EMO_jpt[:, i]/prob_AU[i]))
    
    # 建立由EMO指向AU的概率图
    EMO2_AU = []
    for i in AU:
        EMO2_AU.append(('EMO', 'AU'+str(i)))
    AU_EMO_model = BayesianNetwork(EMO2_AU)
    # EMO的条件概率，由于EMO没有父亲节点，故值为EMO的边缘概率
    EMO_cpd = TabularCPD(variable='EMO', variable_card=len(EMO), values=[[1.0/len(EMO)]]*len(EMO))
    AU_EMO_model.add_cpds(EMO_cpd)
    # 对于每一个AU
    AU_cpd = {}
    for i, j in enumerate(AU):
        var = 'AU'+str(j)
        AU_cpd[j] = TabularCPD(variable=var, variable_card=2, 
                               values=[[1-value for value in list(EMO2AU_cpt[:, i])], list(EMO2AU_cpt[:, i])],
                               evidence=['EMO'], evidence_card=[len(EMO)]
                               ) # P(AU | EMO)
        AU_EMO_model.add_cpds(AU_cpd[j])

    if AU_EMO_model.check_model(): #确认模型是否正确
        print('The established AU-EMO PGM is Valid') 
    else:
        raise NotImplementedError('The established AU-EMO PGM is NOT valid')

    return AU_EMO_model

if __name__ == '__main__':
    # EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()

    # AU_EMO_model = AU_EMO_bayesGraph(EMO2AU_cpt, prob_AU, EMO, AU)

    AU_EMO_model = initGraph()

'''
import AU_EMO_bayes
from AU_EMO_bayes import initGraph
from InferPGM.inference import VariableElimination

from materials.process_priori import cal_interAUPriori

if __name__ == '__main__':
    AU_evidence = { 'AU1':1, 
                    'AU2':0,
                    'AU4':1,
                    'AU5':0,
                    'AU6':0,
                    'AU7':1,
                    'AU9':1,
                    'AU10':0,
                    'AU11':0,
                    'AU12':0,
                    'AU15':0,
                    'AU17':0,
                    'AU20':0,
                    'AU23':0,
                    'AU24':0,
                    'AU25':1,
                    'AU26':1 }
    
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    AU_EMO_model = AU_EMO_bayes.AU_EMO_bayesGraph(EMO2AU_cpt, prob_AU, EMO, AU)
    q = AU_EMO_model.infer(AU_evidence)

    # AU_EMO_model = initGraph()
    # model_infer = VariableElimination(AU_EMO_model)
    # q = model_infer.query(variables=['EMO'], evidence=AU_evidence)


    print(q)
'''
