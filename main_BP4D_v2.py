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
from losses import *
from utils import *
from conf import parser2dict, get_config,set_logger,set_outdir,set_env

from simclr.modules import NT_Xent


# Train
def train(conf, net, train_loader, optimizer, epoch, criterion_AU, scheduler):
    losses_AU = AverageMeter()
    losses_EMO = AverageMeter()
    losses = AverageMeter()
    accs_EMO = AccAverageMeter()
    criterion_EMO = nn.CrossEntropyLoss()

    net = net.to(device)
    net.train()
    statistics_list = None
    train_loader_len = len(train_loader.dataset)
    num_EMO = len(train_loader.dataset.EMO)
    confu_m = torch.zeros((num_EMO, num_EMO))

    labelsAU_record = []
    labelsEMO_record = []
    predsAU_record = []
    predsEMO_record = []

    for batch_i, (img1, img_SSL, img2, labelsEMO, labelsAU, index) in enumerate(train_loader):
        torch.cuda.empty_cache()
        if img1 is not None:
            labelsAU = labelsAU.float()
            labelsEMO = labelsEMO.reshape(labelsEMO.shape[0])
            if torch.cuda.is_available():
                img1, img2, img_SSL, labelsEMO, labelsAU = img1.to(device), img2.to(device), img_SSL.to(device), labelsEMO.to(device), labelsAU.to(device)
            #-------------------------------train and backward----------------------------
            outputs_AU, output_EMO, hm1 = net(img1)
            outputs_AU_flip, output_EMO_flip, hm2 = net(img2)
            
            loss_AU = criterion_AU(outputs_AU, labelsAU)
            grid_l = generate_flip_grid(device, conf.w, conf.h)
            loss1 = criterion_EMO(output_EMO, labelsEMO)
            flip_loss_l = ACLoss(hm1, hm2, grid_l, output_EMO)
            loss_EMO = loss1 + conf.lam_EMO * flip_loss_l
            loss = loss_AU + loss_EMO
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #-------------------------------train and backward----------------------------

            #-------------------------------loss and accuracy----------------------------
            losses_AU.update(loss_AU.item(), img1.size(0))
            update_list = statistics(outputs_AU, labelsAU.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
            _, predicts_EMO = torch.max(output_EMO, 1)
            correct_num_EMO = torch.eq(predicts_EMO, labelsEMO).sum()
            losses_EMO.update(loss_EMO.item(), img1.size(0))
            accs_EMO.update(correct_num_EMO, img1.size(0))
            confu_m = confusion_matrix(predicts_EMO, labels=labelsEMO, conf_matrix=confu_m)
            losses.update(loss_AU.item() + loss_EMO.item(), img1.size(0))
            #-------------------------------loss and accuracy----------------------------
            # adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate_AU, batch_i, train_loader_len)
            scheduler.step()
            #-------------------------------rules info----------------------------
            labelsAU_record.append(labelsAU)
            outputs_AU = outputs_AU >= 0.5
            predsAU_record.append(outputs_AU.long())
            labelsEMO_record.append(labelsEMO)
            predsEMO_record.append(predicts_EMO)
            #-------------------------------rules info----------------------------

    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    
    train_info_rules = {}
    train_info_rules['labelsAU_record'] = torch.cat(labelsAU_record)
    train_info_rules['predsAU_record'] = torch.cat(predsAU_record)
    train_info_rules['labelsEMO_record'] = torch.cat(labelsEMO_record)
    train_info_rules['predsEMO_record'] = torch.cat(predsEMO_record)
    AU_info = {}
    AU_info['loss'] = losses_AU.avg
    AU_info['mean_f1_score'] = mean_f1_score
    AU_info['f1_score_list'] = f1_score_list
    AU_info['mean_acc'] = mean_acc
    AU_info['acc_list'] = acc_list
    AU_info['statistics_list'] = statistics_list
    train_info_rules['AU_info'] = AU_info
    EMO_info = {}
    EMO_info['loss'] = losses_EMO.avg
    EMO_info['acc'] = accs_EMO.avg
    EMO_info['confu_m'] = confu_m
    train_info_rules['EMO_info'] = EMO_info

    AU_return = (losses_AU.avg, mean_f1_score, f1_score_list, mean_acc, acc_list)
    EMO_return = (losses_EMO.avg, accs_EMO.avg, confu_m)
    # backbone_return = (losses_SSL.avg, losses.avg)
    return losses.avg, AU_return, EMO_return, train_info_rules


def val(net, val_loader, criterion_AU):
    losses_AU = AverageMeter()
    losses_EMO = AverageMeter()
    losses = AverageMeter()
    accs_EMO = AccAverageMeter()
    num_EMO = len(val_loader.dataset.EMO)
    confu_m = torch.zeros((num_EMO, num_EMO))
    criterion_EMO = nn.CrossEntropyLoss()

    net = net.to(device)
    net.eval()
    statistics_list = None
    labelsAU_record = []
    labelsEMO_record = []
    predsAU_record = []
    predsEMO_record = []
    
    for batch_i, (img1, img_SSL, img2, labelsEMO, labelsAU, index) in enumerate(val_loader):
        torch.cuda.empty_cache()
        if img1 is not None:
            with torch.no_grad():
                labelsAU = labelsAU.float()
                labelsEMO = labelsEMO.reshape(labelsEMO.shape[0])
                if torch.cuda.is_available():
                    img1, img2, img_SSL, labelsEMO, labelsAU = img1.to(device), img2.to(device), img_SSL.to(device), labelsEMO.to(device), labelsAU.to(device)
                #-------------------------------train and backward----------------------------
                outputs_AU, output_EMO, hm1 = net(img1)
                outputs_AU_flip, output_EMO_flip, hm2 = net(img2)
                
                loss_AU = criterion_AU(outputs_AU, labelsAU)
                grid_l = generate_flip_grid(device, conf.w, conf.h)
                loss1 = criterion_EMO(output_EMO, labelsEMO)
                flip_loss_l = ACLoss(hm1, hm2, grid_l, output_EMO)
                loss_EMO = loss1 + conf.lam_EMO * flip_loss_l
                loss = loss_AU + loss_EMO
                #-------------------------------train and backward----------------------------

                #-------------------------------loss and accuracy----------------------------
                losses_AU.update(loss_AU.item(), img1.size(0))
                update_list = statistics(outputs_AU, labelsAU.detach(), 0.5)
                statistics_list = update_statistics_list(statistics_list, update_list)
                _, predicts_EMO = torch.max(output_EMO, 1)
                correct_num_EMO = torch.eq(predicts_EMO, labelsEMO).sum()
                losses_EMO.update(loss_EMO.item(), img1.size(0))
                accs_EMO.update(correct_num_EMO, img1.size(0))
                confu_m = confusion_matrix(predicts_EMO, labels=labelsEMO, conf_matrix=confu_m)
                losses.update(loss_AU.item() + loss_EMO.item(), img1.size(0))
                #-------------------------------loss and accuracy----------------------------
                
                #-------------------------------rules info----------------------------
                labelsAU_record.append(labelsAU)
                outputs_AU = outputs_AU >= 0.5
                predsAU_record.append(outputs_AU.long())
                labelsEMO_record.append(labelsEMO)
                predsEMO_record.append(predicts_EMO)
                #-------------------------------rules info----------------------------
                
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)

    val_info_rules = {} 
    val_info_rules['labelsAU_record'] = torch.cat(labelsAU_record)
    val_info_rules['predsAU_record'] = torch.cat(predsAU_record)
    val_info_rules['labelsEMO_record'] = torch.cat(labelsEMO_record)
    val_info_rules['predsEMO_record'] = torch.cat(predsEMO_record)
    AU_info = {}
    AU_info['loss'] = losses_AU.avg
    AU_info['mean_f1_score'] = mean_f1_score
    AU_info['f1_score_list'] = f1_score_list
    AU_info['mean_acc'] = mean_acc
    AU_info['acc_list'] = acc_list
    AU_info['statistics_list'] = statistics_list
    val_info_rules['AU_info'] = AU_info
    EMO_info = {}
    EMO_info['loss'] = losses_EMO.avg
    EMO_info['acc'] = accs_EMO.avg
    EMO_info['confu_m'] = confu_m
    val_info_rules['EMO_info'] = EMO_info
    AU_return = (losses_AU.avg, mean_f1_score, f1_score_list, mean_acc, acc_list)
    EMO_return = (losses_EMO.avg, accs_EMO.avg, confu_m)
    # backbone_return = (losses_SSL.avg, losses.avg)
    return losses.avg, AU_return, EMO_return, val_info_rules


def main(conf):
    #------------------------------Data Preparation--------------------------
    train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
    train_weight_AU = train_loader.dataset.train_weight_AU
    EMO = train_loader.dataset.EMO
    AU = train_loader.dataset.AU
    dataset_info = infolist(EMO, AU)
    conf.num_classesAU = len(AU)
    #------------------------------Data Preparation--------------------------

    #---------------------------------Setting-----------------------------
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, test_len))

    net = TwoBranch_Pred(pretrained=True, num_AUs=conf.num_classesAU, num_EMOs=len(EMO), neighbor_num=conf.neighbor_num, metric=conf.metric, 
                         ispred_AU=True, ispred_EMO=True)
    net = net.to(device)

    train_weight_AU = torch.from_numpy(train_weight_AU).to(device)
    criterion_AU = WeightedAsymmetricLoss(weight=train_weight_AU)
    conf.learning_rate_AU = 0.001
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate_SSL_fine, weight_decay=conf.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #---------------------------------Setting-----------------------------

    #---------------------------------RULE Setting-----------------------------
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
    ori_size = np.sum(np.array(EMO_img_num))
    num_all_img = ori_size
    AU_ij_cnt = AU_cpt * ori_size
    AU_cnt = prob_AU * ori_size
    input_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    dataset_AU = train_loader.dataset.AU
    priori_AU = train_loader.dataset.priori['AU']
    AU_p_d = (priori_AU, dataset_AU)
    source_list = [
                ['predsAU_record', 'labelsEMO_record'],
                ['labelsAU_record', 'predsEMO_record'],
                ['labelsAU_record', 'labelsEMO_record'],
                ['predsAU_record', 'predsEMO_record']
                ]
    #---------------------------------RULE Setting-----------------------------

    logging.info('the init learning rate of EMO, AU and rules are {}, {}, {}'
                .format(conf.learning_rate_EMO, conf.learning_rate_AU, conf.lr_relation))

    for epoch in range(conf.start_epoch, conf.epochs):
        lr_TB = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR_TwoBranch: {}, LR_RULES: {} ]".format(epoch + 1, conf.epochs, lr_TB, conf.lr_relation))
        
        train_losses, train_info_AU, train_info_EMO, train_input_info = train(conf, net, train_loader, optimizer, epoch, criterion_AU, scheduler)
        train_AU_loss, train_AU_mean_f1_score, train_AU_f1_score, train_AU_mean_acc, train_AU_acc = train_info_AU
        train_EMO_loss, train_EMO_acc, train_confuse_EMO = train_info_EMO

        val_losses, val_AU_return, val_EMO_return, val_input_info = val(net, test_loader, criterion_AU)
        val_AU_loss, val_AU_mean_f1_score, val_AU_f1_score, val_AU_mean_acc, val_AU_acc = val_AU_return
        val_EMO_loss, val_EMO_acc, val_confuse_EMO = val_EMO_return
        
        #---------------------------------Logging Part--------------------------
        # AUlog
        infostr_AU = {'Epoch: {} train_AU_loss: {:.5f} val_AU_loss: {:.5f} val_AU_mean_f1_score {:.2f} val_AU_mean_acc {:.2f}'
                .format(epoch + 1, train_AU_loss, val_AU_loss, 100.* val_AU_mean_f1_score, 100.* val_AU_mean_acc)}
        logging.info(infostr_AU)
        infostr_AU = {'AU Val F1-score-list:'}
        logging.info(infostr_AU)
        infostr_AU = dataset_info.info_AU(val_AU_f1_score)
        logging.info(infostr_AU)
        infostr_AU = {'AU Val Acc-list:'}
        logging.info(infostr_AU)
        infostr_AU = dataset_info.info_AU(val_AU_acc)
        logging.info(infostr_AU)

        # EMOlog
        infostr_EMO = {'Epoch {} train_EMO_loss: {:.5f} train_EMO_acc: {:.2f} val_EMO_loss: {:.5f} val_EMO_acc: {:.2f}'
                        .format(epoch + 1, train_EMO_loss, 100.* train_EMO_acc, val_EMO_loss, 100.* val_EMO_acc)}
        logging.info(infostr_EMO)
        infostr_EMO = {'EMO Val Acc-list:'}
        logging.info(infostr_EMO)
        for i in range(val_confuse_EMO.shape[0]):
            val_confuse_EMO[:, i] = val_confuse_EMO[:, i] / val_confuse_EMO[:, i].sum(axis=0)
        infostr_EMO = dataset_info.info_EMO(torch.diag(val_confuse_EMO).cpu().numpy().tolist())
        logging.info(infostr_EMO)
        #---------------------------------Logging Part--------------------------
        
        if (epoch+1) % conf.save_epoch == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_input_info': train_input_info,
                'val_input_info': val_input_info,
                'input_rules': input_rules
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))
            
            for info_source in source_list:
                f = 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'
                logging.info('The current info source are {} and {}, the features are from {} '
                    .format(info_source[0], info_source[1], f))
                train_rules_loss, train_rules_acc, train_confu_m, val_rules_loss, val_rules_acc, val_confu_m = main_rules(conf, device, f, info_source, AU_p_d)
                
                infostr_rules = {'Epoch {} train_rules_loss: {:.5f}, train_rules_acc: {:.2f}, val_rules_loss: {:.5f}, val_rules_acc: {:.2f}'
                                    .format(epoch+1, train_rules_loss, 100.* train_rules_acc, val_rules_loss, 100.* val_rules_acc)}
                logging.info(infostr_rules)
                infostr_EMO = {'EMO Rules Train Acc-list:'}
                logging.info(infostr_EMO)
                for i in range(train_confu_m.shape[0]):
                    train_confu_m[:, i] = train_confu_m[:, i] / train_confu_m[:, i].sum(axis=0)
                infostr_EMO = dataset_info.info_EMO(torch.diag(train_confu_m).cpu().numpy().tolist())
                logging.info(infostr_EMO)
                
                infostr_EMO = {'EMO Rules Val Acc-list:'}
                logging.info(infostr_EMO)
                for i in range(val_confu_m.shape[0]):
                    val_confu_m[:, i] = val_confu_m[:, i] / val_confu_m[:, i].sum(axis=0)
                infostr_EMO = dataset_info.info_EMO(torch.diag(val_confu_m).cpu().numpy().tolist())
                logging.info(infostr_EMO)
                del train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc

    shutil.copyfile(os.path.join(conf['outdir'], 'train.log'), os.path.join(conf['outdir'], 'train_copy.log'))
    end_flag = 1

if __name__=='__main__':
    conf = parser2dict()
    conf.dataset = 'BP4D'
    conf = get_config(conf)
    conf.gpu = 3
    conf.exp_name = 'Test_v2'
    conf.learning_rate_SSL_fine = 0.1

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
