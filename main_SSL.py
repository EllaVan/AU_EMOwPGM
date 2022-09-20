import os
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import pickle as pkl
import time
from tensorboardX import SummaryWriter

from models.TwoBranch import GraphAU, EAC
from models.TwoBranch_v2 import TwoBranch_Pred
from models.rules_BP4D import *
from models.resnet import *
from losses import *
from utils import *
from conf import parser2dict, get_config,set_logger,set_outdir,set_env

from simclr.modules import NT_Xent
from simclr import SimCLR
from simclr.modules.identity import Identity


# Train
def train(conf, net, train_loader, optimizer, epoch, criterion_SSL):
    losses_SSL = AverageMeter()

    net = net.to(device)
    net.train()
    train_loader_len = len(train_loader.dataset)
    for batch_i, (img1, img_SSL, img2, labelsEMO, labelsAU, index) in enumerate(train_loader):
        torch.cuda.empty_cache()
        if img1 is not None:
            labelsAU = labelsAU.float()
            labelsEMO = labelsEMO.reshape(labelsEMO.shape[0])
            if torch.cuda.is_available():
                img1, img2, img_SSL, labelsEMO, labelsAU = img1.to(device), img2.to(device), img_SSL.to(device), labelsEMO.to(device), labelsAU.to(device)
            #-------------------------------train and backward----------------------------
            h_i, h_j, z_i, z_j = net(img1, img_SSL)

            loss_SSL = criterion_SSL(z_i, z_j, labelsAU.shape[0])
            loss_SSL.backward()
            optimizer.step()
            #-------------------------------train and backward----------------------------

            #-------------------------------loss and accuracy----------------------------
            losses_SSL.update(loss_SSL.item(), img1.size(0))
            #-------------------------------loss and accuracy----------------------------

            adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate_AU, batch_i, train_loader_len)
    return losses_SSL.avg

def main(conf):
    #------------------------------Data Preparation--------------------------
    train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
    #------------------------------Data Preparation--------------------------

    #---------------------------------Setting-----------------------------
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, test_len))

    encoder = ResNet_EMO(BasicBlock, [2, 2, 2, 2])
    projection_dim = 64
    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer
    net = SimCLR(encoder, projection_dim, n_features)
    net = net.to(device)

    criterion_SSL = NT_Xent(conf.batch_size, conf.SSL_temperature, conf.world_size)
    learning_rate = 0.3 * conf.batch_size / 256
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #---------------------------------Setting-----------------------------

    for epoch in range(conf.start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        
        train_losses = train(conf, net, train_loader, optimizer, epoch, criterion_SSL)
        
        #---------------------------------Logging Part--------------------------
        # AUlog
        infostr_loss = {'Epoch: {} loss: {:.5f} ' .format(epoch + 1, train_losses)}
        logging.info(infostr_loss)
        if (epoch+1) % conf.save_epoch == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))

    shutil.copyfile(os.path.join(conf['outdir'], 'train.log'), os.path.join(conf['outdir'], 'train_copy.log'))
    end_flag = 1

if __name__=='__main__':
    conf = parser2dict()
    conf.dataset = 'BP4D'
    conf = get_config(conf)
    conf.gpu = 1
    conf.exp_name = 'SSL2'
    conf.epochs = 200
    conf.save_epoch = 20

    global device
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    conf.num_gpus = torch.cuda.device_count()
    # conf.world_size = conf.num_gpus * conf.SSL_nodes
    conf.world_size = 1 * conf.SSL_nodes

    set_env(conf)
    set_outdir(conf) # generate outdir name
    set_logger(conf) # Set the logger
    main(conf)
