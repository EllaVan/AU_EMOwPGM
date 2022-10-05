import os
cur_path = os.path.abspath(__file__)
cur_path = os.path.dirname(cur_path)
import sys
sys.path.append(cur_path)
import loss_reweighting as loss_expect
import torch
import torch.nn as nn
from torch.autograd import Variable

from schedule import lr_setter


def weight_learner(cfeatures, model, args, global_epoch=0, iter=0):
    '''
    cfeatures: 样本特征
    pre_features: 来自model的register_buffer, 为全0
    pre_weight1: 来自model的register_buffer, 为全1
    '''
    # pre_features = model.module.pre_features
    # pre_weight1 = model.module.pre_weight1
    pre_features = model.backbone.pre_features
    pre_weight1 = model.backbone.pre_weight1

    device = args.device

    softmax = nn.Softmax(0) # 按列softmax (dim=0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).to(device))
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).to(device))
    cfeaturec.data.copy_(cfeatures.data) # 复制cfeatures
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0) # 将样本特征与pre_features concat，从维度上讲，相当于把样本特征用0重复了一遍
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)

    for epoch in range(args.epochb):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()

        lossb = loss_expect.lossb_expect(all_feature, softmax(all_weight), args.num_f, args.sum) # 这里的all_weight相当于1 / (2*batch_size)
        lossp = softmax(weight).pow(args.decay_pow).sum()
        lambdap = args.lambdap * max((args.lambda_decay_rate ** (global_epoch // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lossg = lossb / lambdap + lossp
        if global_epoch == 0:
            lossg = lossg * args.first_step_cons

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if global_epoch == 0 and iter < 10:
        if cfeatures.size()[0] < pre_features.size()[0]:
            pre_features[:cfeatures.size()[0]] = (pre_features[:cfeatures.size()[0]] * iter + cfeatures) / (iter + 1)
            pre_weight1[:cfeatures.size()[0]] = (pre_weight1[:cfeatures.size()[0]] * iter + weight) / (iter + 1)
        else:
            pre_features = (pre_features * iter + cfeatures) / (iter + 1)
            pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * args.presave_ratio + cfeatures * (
                    1 - args.presave_ratio)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * args.presave_ratio + weight * (
                    1 - args.presave_ratio)

    else:
        pre_features = pre_features * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
        pre_weight1 = pre_weight1 * args.presave_ratio + weight * (1 - args.presave_ratio)

    softmax_weight = softmax(weight)

    # model.module.pre_features.data.copy_(pre_features)
    # model.module.pre_weight1.data.copy_(pre_weight1)
    model.backbone.pre_features.data.copy_(pre_features)
    model.backbone.pre_weight1.data.copy_(pre_weight1)

    return softmax_weight, model
