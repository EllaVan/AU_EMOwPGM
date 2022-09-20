import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np

# from models import AU_EMO_bayes
from materials.process_priori import cal_interAUPriori
from functools import reduce


class UpdateGraph(nn.Module):
    def __init__(self, device, in_channels=1, out_channels=6, W=None, prob_all_au=None, init=None):
        super(UpdateGraph, self).__init__()
        # self.device = conf.device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init = init
        self.W = torch.from_numpy(W).float()
        if prob_all_au is None:
            prob_all_au = [[1]] * W.shape[0]
        self.prob_all_au = torch.from_numpy(prob_all_au).float() # torch.from_numpy(prob_all_au, dtype=torch.float32)

        if self.init is None:
            self.init = torch.ones((self.in_channels, self.out_channels))
        else:
            self.init = torch.from_numpy(self.init).float()
        for i in range(self.W.shape[1]):
            for j in range(self.W.shape[0]):
                self.init[:, i] = self.init[:, i] * self.W[j][i]*self.prob_all_au[j]
        self.d1 = F.normalize(self.init, p = 1, dim=1)
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.fc.weight = Parameter(self.d1.T)

    def forward(self, x):
        self.x = x
        self.out = self.fc(self.x)
        return self.out


if __name__ == '__main__':
    # AU_evidence = torch.ones((1, 1)).cuda()
    # EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    # AU_EMO_model = AU_EMO_bayes.AU_EMO_bayesGraph(EMO2AU_cpt, prob_AU, EMO, AU)
    # q = AU_EMO_model.infer(AU_evidence)

    # update_graph = UpdateGraph(AU, AU_evidence, EMO2AU_cpt)
    # q1 = update_graph()

    from functools import reduce
    a = np.array(range(1, 10))
    a = a.reshape((3, 3))
    c = a[:, 0].tolist()
    b = reduce(lambda x, y: x *y, c)

    end_flag = True
