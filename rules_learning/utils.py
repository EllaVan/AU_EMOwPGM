import sys
sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import itertools

import torch

from rules_learning.dataset_process import info_load


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_to_float(loss):
    if isinstance(loss, torch.Tensor):
        return loss.item()
    else:
        return float(loss)

def getDatasetInfo(dataset):
    if dataset == 'BP4D':
        train_loader, test_loader = info_load.getBP4Ddata()
    if dataset == 'CASME':
        train_loader, test_loader = info_load.getCASMEdata()
    return train_loader, test_loader

def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds.reshape(1, -1), labels.reshape(1, -1)):
        conf_matrix[p, t] += 1
    return conf_matrix

'''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
'''
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm2 = cm.copy()
    if normalize:
        for i in range(cm.shape[0]):
            cm2[:, i] = cm[:, i] / cm[:, i].sum(axis=0)
        # cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm2)
    plt.imshow(cm2, interpolation='nearest', cmap="YlGnBu")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
	# 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
	# x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
	# 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm2.max() / 2.
    for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
        # num = float('{:.2f}'.format(cm[i, j])) if normalize else int(cm[i, j])
        num = cm2[i, j]
        plt.text(i, j, '{:.3f}'.format(cm2[i, j]),
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num >= thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    save_path = os.path.join('/media/data1/wf/AU_EMOwPGM/codes/save/CASME/statistics', title+'.jpg')
    plt.savefig(save_path, dpi=500)
    plt.show()
