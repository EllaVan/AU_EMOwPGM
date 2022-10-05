import sys
# sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')

import random

from math import cos, pi
import torch
from torch.utils.data import DataLoader

from materials.dataset import *

# from conf import get_config

# color_jitter = transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0)
# train_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomResizedCrop(size=(224, 224)),
#     transforms.RandomHorizontalFlip(),  # with 0.5 probability
#     transforms.RandomApply([color_jitter], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),])
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.4,
                            contrast=0.4,
                            saturation=0.4,
                            hue=0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
eval_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                            ])

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

def getDatasetInfo(conf):
    if conf.dataset == 'BP4D' or conf.dataset == 'CASME':
        train_dataset = BP4D(conf.dataset_path, phase='train', fold=conf.fold, transform=train_transforms)
        train_len = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=True)
        # train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), shuffle=False,num_workers=conf.num_workers, pin_memory=True)
        test_dataset = BP4D(conf.dataset_path, phase='test', fold=conf.fold, transform=eval_transforms)
        test_len = len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
    elif conf.dataset == 'RAF-DB' or conf.dataset == 'RAF-DB_compound' or conf.dataset == 'AffectNet':
        train_dataset = RAF(conf.dataset_path, phase='train', fold=conf.fold, transform=train_transforms)
        train_len = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), shuffle=False,num_workers=conf.num_workers, pin_memory=True)
        test_dataset = RAF(conf.dataset_path, phase='test', fold=conf.fold, transform=eval_transforms)
        test_len = len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)
    elif conf.dataset == 'DISFA':
        train_dataset = DISFA(conf.dataset_path, phase='train', fold=conf.fold, transform=train_transforms)
        train_len = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=True)
        test_dataset = DISFA(conf.dataset_path, phase='test', fold=conf.fold, transform=eval_transforms)
        test_len = len(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, pin_memory=True)

    return train_loader, test_loader, train_len, test_len#, train_rule_loader

def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model

class infolist():
    def __init__(self, EMO=None, AU=None):
        super(infolist, self).__init__
        self.AU = AU
        self.EMO = EMO

        if self.AU is not None:
            infostr_AU = 'AU' + str(AU[0]) + ': {0[0]:.2f}'
            for i in range(1, len(AU)):
                tmp = 'AU' + str(AU[i]) + ': {0['+str(i)+']:.2f}'
                infostr_AU = infostr_AU + ', ' + tmp
            self.infostr_AU = infostr_AU
        if self.EMO is not None:
            infostr_EMO = EMO[0] + ': {0[0]:.2f}'
            for i in range(1, len(EMO)):
                tmp = EMO[i] + ': {0['+str(i)+']:.2f}'
                infostr_EMO = infostr_EMO + ', ' + tmp
            self.infostr_EMO = infostr_EMO
    
    def info_AU(self, list):
        list_value = [i * 100 for i in list]
        return {self.infostr_AU.format(list_value)}

    def info_EMO(self, list):
        list_value = [i * 100 for i in list]
        return {self.infostr_EMO.format(list_value)}

def BP4D_infolistAU(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU17: {:.2f} AU23: {:.2f} AU24: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9],100.*list[10],100.*list[11])}
    return infostr

def DISFA_infolistAU(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7])}
    return infostr

def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):

    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_rules_lr(init_lr, iteration, num_iter):
    current_iter = iteration
    max_iter = num_iter
    lr = init_lr * (1 + cos(pi * (1.0/2) * current_iter / max_iter)) / 2 # * (1.0/2)
    return lr

def adjust_rules_lr_v2(optimizer, init_lr, iteration, num_iter):
    current_iter = iteration
    max_iter = num_iter
    lr = init_lr * (1 + cos(pi * (1.0/2) * current_iter / max_iter)) / 2 # * (1.0/2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AccAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.avg = 0
        self.sum = 0

    def update(self, val_correct, val_sum):
        self.correct += val_correct
        self.sum += val_sum
        self.avg = self.correct / self.sum


def statistics(pred, y, thresh):
    batch_size = pred.size(0)
    class_nb = pred.size(1)
    pred = pred >= thresh
    pred = pred.long()
    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    assert False
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    assert False
            else:
                assert False
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list


def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list


def calc_acc(statistics_list):
    acc_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        TN = statistics_list[i]['TN']

        acc = (TP+TN)/(TP+TN+FP+FN)
        acc_list.append(acc)
    mean_acc_score = sum(acc_list) / len(acc_list)

    return mean_acc_score, acc_list


def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    assert len(old_list) == len(new_list)

    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']

    return old_list

def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def generate_flip_grid(device, w, h):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid

def walkFile(file):
    file_list = []
    for root, dirs, files in os.walk(file): #root 表示当前正在访问的文件夹路径, dirs 表示该文件夹下的子目录名list, files 表示该文件夹下的文件list
        for f in files: # 遍历文件
            if f.split('.')[-1] == 'pth':
                file_list.append(f)
        break
        # for d in dirs: # 遍历所有的文件夹
            # print(os.path.join(root, d))
    return file_list




# if __name__=='__main__':
#     conf = get_config()
#     train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)

#     for batch_i, (img1, img2, labelsEMO, labelsAU, index) in enumerate(train_loader):
#         # pass
#         torch.cuda.empty_cache()
