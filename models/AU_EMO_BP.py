import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('/media/database/data4/wf/AU_EMOwPGM/codes')

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np

from models import AU_EMO_bayes
from materials.process_priori import cal_interAUPriori

'''
+--------+------------+
| EMO    |   phi(EMO) |
+========+============+
| EMO(0) |     0.1612 |
+--------+------------+
| EMO(1) |     0.1564 |
+--------+------------+
| EMO(2) |     0.1860 |
+--------+------------+
| EMO(3) |     0.1679 |
+--------+------------+
| EMO(4) |     0.1741 |
+--------+------------+
| EMO(5) |     0.1545 |
+--------+------------+
'''

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


class UpdateGraph(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, W=None):
        super(UpdateGraph, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.tensor(W, dtype=torch.float32)
        
        self.d = torch.zeros((self.in_channels, self.out_channels))
        for i in range(self.W.shape[1]):
            self.d[:, i] = 1
            for j in range(self.W.shape[0]):
                self.d[:, i] *= self.W[j][i]
        self.d = F.normalize(self.d, p = 1, dim=1)
        self.d = Parameter(self.d, requires_grad=True)

        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.fc.weight = Parameter(self.d.T)

    def forward(self, x):
        self.x = x
        self.out = self.fc(self.x)
        return self.out


if __name__ == '__main__':
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    AU_EMO_model = AU_EMO_bayes.AU_EMO_bayesGraph(EMO2AU_cpt, prob_AU, EMO, AU)
    q = AU_EMO_model.infer(AU_evidence)

    update_graph = UpdateGraph(AU, AU_evidence, EMO2AU_cpt)
    q1 = update_graph()

    end_flag = True
