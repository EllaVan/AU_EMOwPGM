import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


class Temp(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, W=None):
        super(Temp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.tensor(W, dtype=torch.float32)
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.fc.weight = Parameter(self.W.T)

    def forward(self, x):
        self.x = x
        self.out = self.fc(self.x)
        # self.out_prob = F.normalize(self.out, p = 1, dim=1)

        return self.out


class Temp2(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, W=None):
        super(Temp2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Parameter(W, requires_grad=True)
        
        self.d = torch.zeros((self.in_channels, self.out_channels))
        for i in range(self.W.shape[1]):
            self.d[:, i] = 1
            for j in range(self.W.shape[0]):
                self.d[:, i] *= self.W[j][i]
        self.c = F.normalize(self.d, p = 1, dim=1)
        self.d = Parameter(self.d, requires_grad=True)

        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.fc.weight = Parameter(self.d.T)
        print(self.fc.weight.T)


    def forward(self, x):
        self.x = x
        self.out = self.fc(self.x)
        # self.out_prob = F.normalize(self.out, p = 1, dim=1)

        return self.out


if __name__ == '__main__':
    label = np.array([3])
    label = torch.tensor(label)
    a = np.array([[1, 2, 3, 4], [1, 2, 1, 1]])/10.0
    b = torch.tensor(a, dtype=torch.float32)
    # print(b)
    # c = []
    # for i in range(b.shape[1]):
    #     prob = 1
    #     for j in range(b.shape[0]):
    #         prob *= b[j][i]
    #     c.append(prob)
    # c = torch.tensor(c, dtype=torch.float32).unsqueeze(0)
    # d1 = F.normalize(c, p = 1, dim=1)

    temp = Temp2(W=b)
    x = torch.ones((1, 1))
    out = temp(x)

    CE = nn.CrossEntropyLoss()
    err = CE(out, label)
    # optimizer = optim.SGD(
    #     [{'params': temp.parameters(), 'lr': 0.1},
    #     {'params': temp.W, 'lr': 0.1}])
    optimizer = optim.SGD(temp.parameters(), lr=0.1)
    optimizer.zero_grad()
    err.backward()
    optimizer.step()
    # print(temp.W)
    # print(temp.fc.weight.T)
    for param in temp.named_parameters():
        print(param)

    end_flag = True