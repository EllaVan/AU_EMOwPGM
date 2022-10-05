import sys
# from tkinter import Variable
from torch.autograd import Variable
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
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.d1 = F.normalize(self.init, p = 1, dim=1)
        self.fc.weight = Parameter(self.d1.T)

    def forward(self, x):
        self.x = x
        self.out = self.fc(self.x)
        return self.out

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        ctx.save_for_backward(input_)    # 将输入保存起来，在backward时使用
        output = input_.clamp(min=1e-5)               # relu就是截断负数，让所有负数等于0
        output = output.clamp(max=1) 
        return output
    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu
        input_,  = ctx.saved_tensors
        grad_input = grad_output.clone() #**就是上面推导的 σ（l+1）**
        grad_input[input_ < 1e-5] = 0                # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0   **zl<=0**
        grad_input[input_ > 1] = 0
        return grad_input
def relu(input_):
    # MyReLU()是创建一个MyReLU对象，
    # Function类利用了Python __call__操作，使得可以直接使用对象调用__call__制定的方法
    # __call__指定的方法是forward，因此下面这句MyReLU（）（input_）相当于
    # return MyReLU().forward(input_)
    return MyReLU().apply(input_)

class UpdateGraph_v2(nn.Module):
    def __init__(self, conf, EMO2AU_cpt, prob_AU, loc1, loc2):
        super(UpdateGraph_v2, self).__init__()
        self.conf = conf
        self.loc1 = loc1
        self.loc2 = loc2

        self.register_buffer('prob_AU', torch.from_numpy(prob_AU[loc1]))
        self.register_buffer('static_prob_AU', torch.from_numpy(prob_AU[loc2]))

        EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
        self.EMO2AU_cpt = Parameter(Variable(torch.from_numpy(EMO2AU_cpt[:, loc1])))
        self.EMO2AU_cpt.requires_grad = True

        self.register_buffer('static_EMO2AU_cpt', torch.from_numpy(EMO2AU_cpt[:, loc2]))
        self.register_buffer('neg_static_EMO2AU_cpt', torch.from_numpy(1 - EMO2AU_cpt[:, loc2]))

    def get_mask(self, prob_all_au, EMO2AU):
        occ_pos = torch.where(prob_all_au > 0.6)[0]
        occ_mask1 = torch.zeros_like(prob_all_au).cuda()
        neg_mask1 = torch.ones_like(prob_all_au).cuda()
        occ_mask1[occ_pos, :] = 1
        neg_mask1[occ_pos, :] = 0
        occ_mask2 = occ_mask1.reshape(1, -1).repeat(EMO2AU.shape[0], 1).cuda()
        neg_mask2 = neg_mask1.reshape(1, -1).repeat(EMO2AU.shape[0], 1).cuda()
        return occ_pos, occ_mask1, neg_mask1, occ_mask2, neg_mask2

    def forward(self, prob_all_au):
        loc1 = self.loc1
        loc2 = self.loc2
        conf = self.conf
        
        self.neg_EMO2AU_cpt = 1 - self.EMO2AU_cpt
        self.neg_EMO2AU_cpt = torch.where(self.neg_EMO2AU_cpt > 0, self.neg_EMO2AU_cpt, conf.zeroPad)
        self.prob_all_au = torch.from_numpy(prob_all_au[loc1, :]).cuda()
        self.static_prob_all_au = torch.from_numpy(prob_all_au[loc2, :]).cuda()

        occ_pos, occ_mask1, neg_mask1, occ_mask2, neg_mask2 = self.get_mask(self.prob_all_au, self.EMO2AU_cpt)
        EMO2AU_weight = occ_mask2 * self.EMO2AU_cpt + neg_mask2 * self.neg_EMO2AU_cpt
        AU_weight = occ_mask1.reshape(self.prob_AU.shape) / self.prob_AU * self.prob_all_au.reshape(self.prob_AU.shape) + neg_mask1.reshape(self.prob_AU.shape) / self.prob_AU
        AU_weight = AU_weight.reshape(1, -1).repeat(EMO2AU_weight.shape[0], 1)
        weight1 = EMO2AU_weight * AU_weight

        occ_pos, occ_mask1, neg_mask1, occ_mask2, neg_mask2 = self.get_mask(self.static_prob_all_au, self.static_EMO2AU_cpt)
        EMO2AU_weight = occ_mask2 * self.static_EMO2AU_cpt + neg_mask2 * self.neg_static_EMO2AU_cpt
        AU_weight = occ_mask1.reshape(self.static_prob_AU.shape) / self.static_prob_AU + neg_mask1.reshape(self.static_prob_AU.shape) / self.static_prob_AU
        AU_weight = AU_weight.reshape(1, -1).repeat(EMO2AU_weight.shape[0], 1)
        weight2 = EMO2AU_weight * AU_weight

        a = []
        for i in range(weight1.shape[0]):
            prob_emo = torch.prod(weight1[i, :]) * torch.prod(weight2[i, :])
            a.append(prob_emo.reshape(1, -1))
        out1 = torch.cat(a).reshape(-1, weight1.shape[0])
        out2 = F.normalize(out1, p = 1, dim=1)

        return out2

class UpdateGraph_continuous(nn.Module):
    def __init__(self, conf, input_rules, loc1, loc2):
        super(UpdateGraph_continuous, self).__init__()
        self.conf = conf
        self.loc1 = loc1
        self.loc2 = loc2

        self.input_rules = input_rules
        EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules

        self.register_buffer('prob_AU', torch.from_numpy(prob_AU[loc1]))
        self.register_buffer('static_prob_AU', torch.from_numpy(prob_AU[loc2]))

        EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
        self.EMO2AU_cpt = Parameter(Variable(torch.from_numpy(EMO2AU_cpt[:, loc1])))
        self.EMO2AU_cpt.requires_grad = True

        self.register_buffer('static_EMO2AU_cpt', torch.from_numpy(EMO2AU_cpt[:, loc2]))
        self.register_buffer('neg_static_EMO2AU_cpt', torch.from_numpy(1 - EMO2AU_cpt[:, loc2]))

        self.AU_cpt = AU_cpt
        self.num_all_img = num_all_img
        self.AU_ij_cnt = AU_ij_cnt
        self.AU_cnt = AU_cnt
        self.EMO = EMO
        self.AU = AU


    def get_mask(self, prob_all_au, EMO2AU):
        occ_pos = torch.where(prob_all_au > 0.6)[0]
        occ_mask1 = torch.zeros_like(prob_all_au).cuda()
        neg_mask1 = torch.ones_like(prob_all_au).cuda()
        occ_mask1[occ_pos, :] = 1
        neg_mask1[occ_pos, :] = 0
        occ_mask2 = occ_mask1.reshape(1, -1).repeat(EMO2AU.shape[0], 1).cuda()
        neg_mask2 = neg_mask1.reshape(1, -1).repeat(EMO2AU.shape[0], 1).cuda()
        return occ_pos, occ_mask1, neg_mask1, occ_mask2, neg_mask2

    def forward(self, prob_all_au, weight2=None):
        loc1 = self.loc1
        loc2 = self.loc2
        conf = self.conf
        
        self.neg_EMO2AU_cpt = 1 - self.EMO2AU_cpt
        self.neg_EMO2AU_cpt = torch.where(self.neg_EMO2AU_cpt > 0, self.neg_EMO2AU_cpt, conf.zeroPad)
        self.prob_all_au = torch.from_numpy(prob_all_au[loc1, :]).cuda()
        self.static_prob_all_au = torch.from_numpy(prob_all_au[loc2, :]).cuda()

        occ_pos, occ_mask1, neg_mask1, occ_mask2, neg_mask2 = self.get_mask(self.prob_all_au, self.EMO2AU_cpt)
        EMO2AU_weight = occ_mask2 * self.EMO2AU_cpt + neg_mask2 * self.neg_EMO2AU_cpt
        AU_weight = occ_mask1.reshape(self.prob_AU.shape) / self.prob_AU * self.prob_all_au.reshape(self.prob_AU.shape) + neg_mask1.reshape(self.prob_AU.shape) / self.prob_AU
        AU_weight = AU_weight.reshape(1, -1).repeat(EMO2AU_weight.shape[0], 1)
        weight1 = EMO2AU_weight * AU_weight

        self.weight2 = weight2
        if weight2 is None:
            occ_pos, occ_mask1, neg_mask1, occ_mask2, neg_mask2 = self.get_mask(self.static_prob_all_au, self.static_EMO2AU_cpt)
            EMO2AU_weight = occ_mask2 * self.static_EMO2AU_cpt + neg_mask2 * self.neg_static_EMO2AU_cpt
            AU_weight = occ_mask1.reshape(self.static_prob_AU.shape) / self.static_prob_AU + neg_mask1.reshape(self.static_prob_AU.shape) / self.static_prob_AU
            AU_weight = AU_weight.reshape(1, -1).repeat(EMO2AU_weight.shape[0], 1)
            self.weight2 = EMO2AU_weight * AU_weight

        a = []
        for i in range(weight1.shape[0]):
            prob_emo = torch.prod(weight1[i, :]) * torch.prod(self.weight2[i, :])
            a.append(prob_emo.reshape(1, -1))
        out1 = torch.cat(a).reshape(-1, weight1.shape[0])
        self.out2 = F.normalize(out1, p = 1, dim=1)

        return self.out2, self.weight2



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
