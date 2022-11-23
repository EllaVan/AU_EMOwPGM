'''
首先, 明确一下loss函数的输入:

一个pred, shape为 (bs, num_classes), 并且未经过softmax;

一个target, shape为 (bs), 也就是一个向量, 并且未经过one_hot编码。

通过前面的公式可以得出, 我们需要在loss实现是做三件事情:

找到当前batch内每个样本对应的类别标签, 然后根据预先设置好的alpha值给每个样本分配类别权重
计算当前batch内每个样本在类别标签位置的softmax值, 作为公式里的, 因为不管是focal loss还是cross_entropy_loss, 每个样本的n个概率中不属于真实类别的都是用不到的
计算原始的cross_entropy_loss, 但不能求平均, 需要得到每个样本的cross_entropy_loss, 因为需要对每个样本施加不同的权重
'''
import numpy as np
import torch
import torch.nn as nn

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表, 三分类中第0类权重0.2, 第1类权重0.3, 第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

if __name__=='__main__':
    alpha1 = [0.1, 0.2, 0.3, 0.4]
    loss_function1 = MultiClassFocalLossWithAlpha(alpha=alpha1)
    alpha2 = [0.25, 0.25, 0.25, 0.25]
    loss_function2 = MultiClassFocalLossWithAlpha(alpha=alpha2)
    input = np.array([[0.1, 0.2, 0.3, 0.4]])
    input = torch.from_numpy(input)
    print(input)
    label1 = torch.from_numpy(np.array(0))
    err1_1 = loss_function1(input, label1)
    err1_2 = loss_function2(input, label1)
    print(err1_1)
    print(err1_2) # 因为loss_function1中0的位置的alpha权重比loss_function2的小，因此err1_1也比err1_2小
    label2 = torch.from_numpy(np.array(3))
    err2_1 = loss_function1(input, label2)
    err2_2 = loss_function2(input, label2)
    print(err2_1)
    print(err2_2) # 因为loss_function1中0的位置的alpha权重比loss_function2的大，因此err1_1也比err1_2大
    end_flag = 1