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

# from models.TwoBranch import GraphAU, EAC
from models.TwoBranch_v2 import GraphAU_SSL as GraphAU
from models.TwoBranch_v2 import EAC_SSL as EAC
from models.rules_DISFA import *
from losses import *
from utils import *
from conf import parser2dict, get_config,set_logger,set_outdir,set_env

def update_net_info(conf):
    conf.net_params_from_path = '/media/data1/wf/AU_EMOwPGM/codes/results/BP4D/Test_v2/subject_independent/bs_64_seed_0_lrEMO_0.01_lrAU_0.1_lr_relation_0.001/epoch20_model_fold0.pth'
    conf.net_params_from_dataset = 'BP4D'
    all_info = torch.load(conf.net_params_from_path, map_location='cpu')
    conf.train_f1_AU = all_info['val_input_info']['AU_info']['mean_f1_score']
    conf.train_acc_EMO = all_info['val_input_info']['EMO_info']['acc']
    return conf

# Train
def train(conf, net_AU, net_EMO, train_loader, optimizer_AU, optimizer_EMO, epoch, criterion_AU, scheduler_EMO):
    losses_AU = AverageMeter()

    net_AU, net_EMO = net_AU.to(device), net_EMO.to(device)
    net_AU.train()
    net_EMO.train()
    statistics_list = None
    train_loader_len = len(train_loader.dataset)

    labelsAU_record = []
    predsAU_record = []
    predsEMO_record = []
    loc2 = [0, 1, 2, 3, 4, 6, 9, 10, 11, 12]

    for batch_i, (img1, img2, labelsEMO, labelsAU, index) in enumerate(train_loader):
        torch.cuda.empty_cache()
        if img1 is not None:
            #-------------------------------AU train----------------------------
            labelsAU = labelsAU.float()
            if torch.cuda.is_available():
                img1, img2, labelsEMO, labelsAU = img1.to(device), img2.to(device), labelsEMO.to(device), labelsAU.to(device)
            outputs_AU = net_AU(img1)
            loss_AU = criterion_AU(outputs_AU, labelsAU)
            optimizer_AU.zero_grad()
            loss_AU.backward()
            optimizer_AU.step()
            adjust_learning_rate(optimizer_AU, epoch, conf.epochs, conf.learning_rate_AU, batch_i, train_loader_len)

            losses_AU.update(loss_AU.item(), img1.size(0))
            update_list = statistics(outputs_AU, labelsAU.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
            #-------------------------------AU train----------------------------
        
            #-------------------------------EMO train----------------------------
            labelsEMO = labelsEMO.reshape(labelsEMO.shape[0])
            outputs_EMO, _ = net_EMO(img1)
            _, predicts_EMO = torch.max(outputs_EMO, 1)
            #-------------------------------EMO train----------------------------
        
            #-------------------------------rules info----------------------------
            labelsAU_record.append(labelsAU)
            outputs_AU = outputs_AU >= 0.5
            predsAU_record.append(outputs_AU.long())
            predsEMO_record.append(predicts_EMO)
            #-------------------------------rules info----------------------------
            
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    
    train_info_rules = {}
    train_info_rules['labelsAU_record'] = torch.cat(labelsAU_record)
    train_info_rules['predsAU_record'] = torch.cat(predsAU_record)
    train_info_rules['predsEMO_record'] = torch.cat(predsEMO_record)
    AU_info = {}
    AU_info['loss'] = losses_AU.avg
    AU_info['mean_f1_score'] = mean_f1_score
    AU_info['f1_score_list'] = f1_score_list
    AU_info['mean_acc'] = mean_acc
    AU_info['acc_list'] = acc_list
    AU_info['statistics_list'] = statistics_list
    train_info_rules['AU_info'] = AU_info
    AU_return = (losses_AU.avg, mean_f1_score, f1_score_list, mean_acc, acc_list)

    return AU_return, train_info_rules


def val(net_AU, net_EMO, val_loader, criterion_AU):
    losses_AU = AverageMeter()

    net_AU, net_EMO = net_AU.to(device), net_EMO.to(device)
    net_AU.eval()
    net_EMO.eval()
    statistics_list = None
    labelsAU_record = []
    predsAU_record = []
    predsEMO_record = []
    # loc2 = [0, 1, 2, 3, 4, 6, 9, 10, 11, 12]
    
    for batch_i, (img1, img2, labelsEMO, labelsAU, index) in enumerate(val_loader):
        torch.cuda.empty_cache()
        if img1 is not None:
            with torch.no_grad():
                #-------------------------------AU val----------------------------
                labelsAU = labelsAU.float()
                if torch.cuda.is_available():
                    img1, img2, labelsEMO, labelsAU = img1.to(device), img2.to(device), labelsEMO.to(device), labelsAU.to(device)
                outputs_AU = net_AU(img1)
                # labelsAU = labelsAU[:, :-2]
                # outputs_AU = outputs_AU[:, loc2]
                loss_AU = criterion_AU(outputs_AU, labelsAU)
                losses_AU.update(loss_AU.item(), img1.size(0))
                update_list = statistics(outputs_AU, labelsAU.detach(), 0.5)
                statistics_list = update_statistics_list(statistics_list, update_list)
                #-------------------------------AU val----------------------------
        
                #-------------------------------EMO val----------------------------
                labelsEMO = labelsEMO.reshape(labelsEMO.shape[0])
                outputs_EMO, _ = net_EMO(img1)
                _, predicts_EMO = torch.max(outputs_EMO, 1)
                #-------------------------------EMO val----------------------------
                
                #-------------------------------rules info----------------------------
                labelsAU_record.append(labelsAU)
                outputs_AU = outputs_AU >= 0.5
                predsAU_record.append(outputs_AU.long())
                predsEMO_record.append(predicts_EMO)
                #-------------------------------rules info----------------------------
                
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)

    val_info_rules = {}
    val_info_rules['labelsAU_record'] = torch.cat(labelsAU_record)
    val_info_rules['predsAU_record'] = torch.cat(predsAU_record)
    val_info_rules['predsEMO_record'] = torch.cat(predsEMO_record)
    AU_info = {}
    AU_info['loss'] = losses_AU.avg
    AU_info['mean_f1_score'] = mean_f1_score
    AU_info['f1_score_list'] = f1_score_list
    AU_info['mean_acc'] = mean_acc
    AU_info['acc_list'] = acc_list
    AU_info['statistics_list'] = statistics_list
    val_info_rules['AU_info'] = AU_info
    AU_return = (losses_AU.avg, mean_f1_score, f1_score_list, mean_acc, acc_list)
    return AU_return, val_info_rules


def main(conf):
    #------------------------------Data Preparation--------------------------
    train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
    train_weight_AU = train_loader.dataset.train_weight_AU#[0, :-2]
    EMO = train_loader.dataset.EMO
    AU = train_loader.dataset.AU
    dataset_info = infolist(EMO, AU)
    conf.num_classesAU = len(AU)
    #------------------------------Data Preparation--------------------------

    #---------------------------------AU Setting-----------------------------
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, test_len))
    net_AU = GraphAU(num_classes=conf.num_classesAU, neighbor_num=conf.neighbor_num, metric=conf.metric)
    # net_AU = GraphAU(num_classes=15, neighbor_num=conf.neighbor_num, metric=conf.metric)

    if conf.resume != '': # resume
        logging.info("Resume form | {} ]".format(conf.resume))
        net_AU = load_state_dict(net_AU, conf.resume)
    # all_info = torch.load(conf.net_params_from_path, map_location='cpu')#['state_dict']
    # state_dict_AU = all_info['state_dict_AU']
    # net_AU.load_state_dict(state_dict_AU)
    net_AU = net_AU.to(device)

    train_weight_AU = torch.from_numpy(train_weight_AU).to(device)
    criterion_AU = WeightedAsymmetricLoss(weight=train_weight_AU)
    optimizer_AU = optim.AdamW(net_AU.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate_AU, weight_decay=conf.weight_decay)
    #---------------------------------AU Setting-----------------------------

    #---------------------------------EMO Setting-----------------------------
    net_EMO = EAC(conf, num_classes=len(EMO))
    all_info = torch.load(conf.net_params_from_path, map_location='cpu')#['state_dict']
    state_dict_EMO = all_info['state_dict_EMO']
    net_EMO.load_state_dict(state_dict_EMO)
    net_EMO = net_EMO.to(device)

    optimizer_EMO = torch.optim.Adam(net_EMO.parameters() , lr=conf.learning_rate_EMO, weight_decay=1e-4)
    scheduler_EMO = torch.optim.lr_scheduler.ExponentialLR(optimizer_EMO, gamma=0.9)
    #---------------------------------EMO Setting-----------------------------

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
                ['labelsAU_record', 'predsEMO_record'],
                ['predsAU_record', 'predsEMO_record']
                ]
    #---------------------------------RULE Setting-----------------------------

    logging.info('the init learning rate of AU and rules are {}, {}'
                .format(conf.learning_rate_AU, conf.lr_relation))

    for epoch in range(conf.start_epoch, conf.epochs):
        lr_AU = optimizer_AU.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR_AU: {}, LR_RULES: {} ]".format(epoch + 1, conf.epochs, lr_AU, conf.lr_relation))
        
        train_info_AU, train_input_info = train(conf, net_AU, net_EMO, train_loader, optimizer_AU, optimizer_EMO, epoch, criterion_AU, scheduler_EMO)
        train_AU_loss, train_AU_mean_f1_score, train_AU_f1_score, train_AU_mean_acc, train_AU_acc = train_info_AU

        val_AU_return, val_input_info = val(net_AU, net_EMO, test_loader, criterion_AU)
        val_AU_loss, val_AU_mean_f1_score, val_AU_f1_score, val_AU_mean_acc, val_AU_acc = val_AU_return
        
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
        #---------------------------------Logging Part--------------------------
        
        if (epoch+1) % conf.save_epoch == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict_AU': net_AU.state_dict(),
                'state_dict_EMO': net_EMO.state_dict(),
                'optimizer_AU': optimizer_AU.state_dict(),
                'optimizer_EMO': optimizer_EMO.state_dict(),
                'scheduler_EMO': scheduler_EMO.state_dict(),
                'train_input_info': train_input_info,
                'val_input_info': val_input_info,
                'input_rules': input_rules
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))
            
            for info_source in source_list:
                f = 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'
                logging.info('The current info source are {} and {}, the features are from {} '
                    .format(info_source[0], info_source[1], f))
                train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc, val_confu_m = main_rules(conf, device, f, info_source, AU_p_d)
                
                infostr_rules = {'Epoch {} train_rules_loss: {:.5f}, train_rules_acc: {:.2f}, val_rules_loss: {:.5f}, val_rules_acc: {:.2f}'
                                    .format(epoch+1, train_rules_loss, 100.* train_rules_acc, val_rules_loss, 100.* val_rules_acc)}
                logging.info(infostr_rules)
                
                infostr_EMO = {'EMO Rules Val Acc-list:'}
                logging.info(infostr_EMO)
                for i in range(val_confu_m.shape[0]):
                    val_confu_m[:, i] = val_confu_m[:, i] / val_confu_m[:, i].sum(axis=0)
                infostr_EMO = dataset_info.info_EMO(torch.diag(val_confu_m).cpu().numpy().tolist())
                logging.info(infostr_EMO)
                del train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc

    shutil.copyfile(os.path.join(conf['outdir'], 'train.log'), os.path.join(conf['outdir'], 'train_copy.log'))

if __name__=='__main__':
    conf = parser2dict()
    conf.dataset = 'DISFA'
    conf = get_config(conf)
    conf = update_net_info(conf)
    conf.gpu = 2

    global device
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device

    set_env(conf)
    set_outdir(conf) # generate outdir name
    set_logger(conf) # Set the logger
    main(conf)

