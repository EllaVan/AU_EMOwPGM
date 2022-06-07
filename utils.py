import random

import torch

from materials import info_load


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
    return train_loader, test_loader