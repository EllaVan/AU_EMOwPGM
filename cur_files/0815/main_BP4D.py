import os
from re import A
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging
import pickle as pkl

from models.TwoBranch import GraphAU, EAC
from models.rules import *
from losses import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


# Train
def train(conf, net_AU, net_EMO, train_loader, optimizer_AU, optimizer_EMO, epoch, criterion_AU, scheduler_EMO):
    losses_AU = AverageMeter()
    losses_EMO = AverageMeter()
    accs_EMO = AverageMeter()
    losses_rules = AverageMeter()
    accs_rules = AverageMeter()

    net_AU, net_EMO = net_AU.cuda(), net_EMO.cuda()
    net_AU.train()
    net_EMO.train()
    train_loader_len = len(train_loader.dataset)

    '''
    lr_relation_flag = 0
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
    ori_size = np.sum(np.array(EMO_img_num))
    num_all_img = ori_size
    AU_ij_cnt = AU_cpt * ori_size
    AU_cnt = prob_AU * ori_size
    '''

    for batch_i, (img1, img2, labelsEMO, labelsAU, index) in enumerate(train_loader):
        # pass
        
        torch.cuda.empty_cache()
        if img1 is not None:
            #-------------------------------AU train----------------------------
            adjust_learning_rate(optimizer_AU, epoch, conf.epochs, conf.learning_rate_AU, batch_i, train_loader_len)
            labelsAU = labelsAU.float()
            if torch.cuda.is_available():
                img1, img2, labelsEMO, labelsAU = img1.cuda(), img2.cuda(), labelsEMO.cuda(), labelsAU.cuda()
            outputs_AU = net_AU(img1)
            loss_AU = criterion_AU(outputs_AU, labelsAU)
            optimizer_AU.zero_grad()
            loss_AU.backward()
            optimizer_AU.step()

            losses_AU.update(loss_AU.item(), img1.size(0))
            #-------------------------------AU train----------------------------

            #-------------------------------EMO train----------------------------
            labelsEMO = labelsEMO.reshape(labelsEMO.shape[0])
            output_EMO, hm1 = net_EMO(img1)
            output_EMO_flip, hm2 = net_EMO(img2)
            grid_l = generate_flip_grid(conf.w, conf.h, device)
            loss1 = nn.CrossEntropyLoss()(output_EMO, labelsEMO)
            flip_loss_l = ACLoss(hm1, hm2, grid_l, output_EMO)
            loss_EMO = loss1 + conf.lam_EMO * flip_loss_l
            optimizer_EMO.zero_grad()
            loss_EMO.backward()
            optimizer_EMO.step()

            _, predicts_EMO = torch.max(output_EMO, 1)
            correct_num_EMO = torch.eq(predicts_EMO, labelsEMO).sum()
            losses_EMO.update(loss_EMO.item(), 1)
            accs_EMO.update(correct_num_EMO.item()/img1.size(0), 1)
            #-------------------------------EMO train----------------------------

            '''
            #-------------------------------rules train----------------------------
            # if epoch == conf.epochs - 1:
            input_info = (labelsAU, predicts_EMO)
            input_rules = (EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU)
            output_rules, output_records = learn_rules(conf, input_info, input_rules, conf.lr_relation)
            EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = output_rules
            err_record, acc_record = output_records
            losses_rules.update(err_record, 1)
            accs_rules.update(acc_record/img1.size(0), 1)
            if num_all_img-ori_size >= conf.lr_decay_idx and lr_relation_flag == 0:
                lr_relation_flag = 1
                conf.lr_relation /= 10.0
            #-------------------------------rules train----------------------------
            '''
        
        scheduler_EMO.step()
        train_info_AU = (losses_AU.avg)
        train_info_EMO = (losses_EMO.avg, accs_EMO.avg)
        # train_info_rules = (losses_rules.avg, accs_rules.avg, output_rules)
    return train_info_AU, train_info_EMO#, train_info_rules


def val(net_AU, net_EMO, val_loader, criterion_AU):
    losses_AU = AverageMeter()
    losses_EMO = AverageMeter()
    accs_EMO = AverageMeter()
    losses_rules = AverageMeter()
    accs_rules = AverageMeter()
    num_EMO = len(val_loader.dataset.EMO)
    confu_m = torch.zeros((num_EMO, num_EMO))

    net_AU, net_EMO = net_AU.cuda(), net_EMO.cuda()
    net_AU.eval()
    net_EMO.eval()
    statistics_list = None
    AU_evidence = torch.ones((1, 1)).cuda()

    lr_relation_flag = 0
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(val_loader.dataset.priori.values())
    ori_size = np.sum(np.array(EMO_img_num))
    num_all_img = ori_size
    AU_ij_cnt = AU_cpt * ori_size
    AU_cnt = prob_AU * ori_size

    for batch_i, (img1, img2, labelsEMO, labelsAU, index) in enumerate(val_loader):
        torch.cuda.empty_cache()
        if img1 is not None:
            with torch.no_grad():
                #-------------------------------AU val----------------------------
                labelsAU = labelsAU.float()
                if torch.cuda.is_available():
                    img1, img2, labelsEMO, labelsAU = img1.cuda(), img2.cuda(), labelsEMO.cuda(), labelsAU.cuda()
                outputs_AU = net_AU(img1)
                loss_AU = criterion_AU(outputs_AU, labelsAU)
                losses_AU.update(loss_AU.item(), img1.size(0))
                update_list = statistics(outputs_AU, labelsAU.detach(), 0.5)
                statistics_list = update_statistics_list(statistics_list, update_list)
                #-------------------------------AU val----------------------------

                #-------------------------------EMO val----------------------------
                labelsEMO = labelsEMO.reshape(labelsEMO.shape[0])
                outputs_EMO, _ = net_EMO(img1)
                loss_EMO = nn.CrossEntropyLoss()(outputs_EMO, labelsEMO)
                _, predicts_EMO = torch.max(outputs_EMO, 1)
                correct_num_EMO = torch.eq(predicts_EMO, labelsEMO).sum()
                losses_EMO.update(loss_EMO.item(), img1.size(0))
                accs_EMO.update(correct_num_EMO.item(), img1.size(0))
                confu_m = confusion_matrix(predicts_EMO, labels=labelsEMO, conf_matrix=confu_m)
                #-------------------------------EMO val----------------------------

                #-------------------------------rules test----------------------------
                # input_info = (labelsAU, predicts_EMO)
                # output_records = test_rules(input_info, input_rules)
                # err_record, acc_record = output_records
                # losses_rules.update(err_record, 1)
                # accs_rules.update(acc_record/img1.size(0), 1)

                input_info = (labelsAU, predicts_EMO)
                input_rules = (EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU)
                output_rules, output_records = learn_rules(conf, input_info, input_rules, conf.lr_relation)
                EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = output_rules
                err_record, acc_record = output_records
                losses_rules.update(err_record, 1)
                accs_rules.update(acc_record/img1.size(0), 1)
                if num_all_img-ori_size >= conf.lr_decay_idx and lr_relation_flag == 0:
                    lr_relation_flag = 1
                    conf.lr_relation /= 10.0

                #-------------------------------rules train----------------------------
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)

    AU_return = (losses_AU.avg, mean_f1_score, f1_score_list, mean_acc, acc_list)
    EMO_return = (losses_EMO.avg, accs_EMO.avg, confu_m)
    # rules_return = (losses_rules.avg, accs_rules.avg)
    rules_return = (losses_rules.avg, accs_rules.avg, output_rules)
    return AU_return, EMO_return, rules_return


def main(conf):
    #------------------------------Data Preparation--------------------------
    # summary_writer = SummaryWriter(conf.outdir)
    train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
    train_weight_AU = train_loader.dataset.train_weight_AU
    EMO = train_loader.dataset.EMO
    AU = train_loader.dataset.AU
    dataset_info = infolist(EMO, AU)
    conf.num_classesAU = len(AU)
    #------------------------------Data Preparation--------------------------

    #---------------------------------AU Setting-----------------------------
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, test_len))
    net_AU = GraphAU(num_classes=conf.num_classesAU, neighbor_num=conf.neighbor_num, metric=conf.metric)

    if conf.resume != '': # resume
        logging.info("Resume form | {} ]".format(conf.resume))
        net_AU = load_state_dict(net_AU, conf.resume)
    if torch.cuda.is_available():
        net_AU = nn.DataParallel(net_AU).cuda()
        train_weight_AU = torch.from_numpy(train_weight_AU).cuda()
    criterion_AU = WeightedAsymmetricLoss(weight=train_weight_AU)
    optimizer_AU = optim.AdamW(net_AU.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate_AU, weight_decay=conf.weight_decay)
    #---------------------------------AU Setting-----------------------------

    #---------------------------------EMO Setting-----------------------------
    net_EMO = EAC(conf, num_classes=len(EMO))
    optimizer_EMO = torch.optim.Adam(net_EMO.parameters() , lr=conf.learning_rate_EMO, weight_decay=1e-4)
    scheduler_EMO = torch.optim.lr_scheduler.ExponentialLR(optimizer_EMO, gamma=0.9)
    #---------------------------------EMO Setting-----------------------------

    print('the init learning rate of EMO, AU and rules are ', conf.learning_rate_EMO, conf.learning_rate_AU, conf.lr_relation)

    for epoch in range(conf.start_epoch, conf.epochs):
        lr_AU = optimizer_AU.param_groups[0]['lr']
        lr_EMO = optimizer_EMO.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR_AU: {}, LR_EMO: {} ]".format(epoch + 1, conf.epochs, lr_AU, lr_EMO))

        '''
        #---------------------------------AU Part-----------------------------
        train_AU_loss = train_AU(conf,net_AU,train_loader,optimizer_AU,epoch,criterion_AU)
        val_AU_loss, val_AU_mean_f1_score, val_AU_f1_score, val_AU_mean_acc, val_AU_acc = val_AU(net_AU, test_loader, criterion_AU)
        #---------------------------------AU Part-----------------------------

        #---------------------------------EMO Part-----------------------------
        train_EMO_acc, train_EMO_loss = train_EMO(conf, net_EMO, train_loader, optimizer_EMO, scheduler_EMO)
        test_EMO_acc, test_EMO_loss, confuse_EMO = val_EMO(net_EMO, test_loader)
        #---------------------------------EMO Part-----------------------------
        '''
        
        # train_info_AU, train_info_EMO, train_info_rules = train(conf, net_AU, net_EMO, train_loader, optimizer_AU, optimizer_EMO, epoch, criterion_AU, scheduler_EMO)
        train_info_AU, train_info_EMO = train(conf, net_AU, net_EMO, train_loader, 
            optimizer_AU, optimizer_EMO, epoch, criterion_AU, scheduler_EMO)
        train_AU_loss = train_info_AU
        train_EMO_loss, train_EMO_acc = train_info_EMO
        # train_rules_loss, train_rules_acc, rules = train_info_rules

        # val_AU_return, val_EMO_return, val_rules_return = val(net_AU, net_EMO, test_loader, criterion_AU, rules)
        val_AU_return, val_EMO_return, val_rules_return = val(net_AU, net_EMO, test_loader, criterion_AU)
        val_AU_loss, val_AU_mean_f1_score, val_AU_f1_score, val_AU_mean_acc, val_AU_acc = val_AU_return
        test_EMO_loss, test_EMO_acc, confuse_EMO = val_EMO_return
        val_rules_loss, val_rules_acc, rules = val_rules_return
        

        #---------------------------------Logging Part--------------------------
        # AUlog
        infostr_AU = {'Epoch: {} train_AU_loss: {:.5f} val_AU_loss: {:.5f} val_AU_mean_f1_score {:.2f} val_AU_mean_acc {:.2f}'
                .format(epoch + 1, train_AU_loss, val_AU_loss, 100.* val_AU_mean_f1_score, 100.* val_AU_mean_acc)}
        logging.info(infostr_AU)
        infostr_AU = {'AU F1-score-list:'}
        logging.info(infostr_AU)
        infostr_AU = dataset_info.info_AU(val_AU_f1_score)
        logging.info(infostr_AU)
        infostr_AU = {'AU Acc-list:'}
        logging.info(infostr_AU)
        infostr_AU = dataset_info.info_AU(val_AU_acc)
        logging.info(infostr_AU)
        '''
        summary_writer.add_scalar('train_AU_loss', train_AU_loss, epoch)
        summary_writer.add_scalar('val_AU_loss', train_AU_loss, epoch)
        summary_writer.add_scalar('val_AU_mean_f1_score', val_AU_mean_f1_score, epoch)
        '''

        # EMOlog
        infostr_EMO = {'Epoch {} train_EMO_acc: {:.5f} train_EMO_loss: {:.5f}  val_EMO_acc: {:.5f} val_EMO_loss: {:.5f}'
                        .format(epoch + 1, train_EMO_acc, train_EMO_loss, test_EMO_acc, test_EMO_loss)}
        logging.info(infostr_EMO)
        infostr_EMO = {'EMO Acc-list:'}
        logging.info(infostr_EMO)
        for i in range(confuse_EMO.shape[0]):
            confuse_EMO[:, i] = confuse_EMO[:, i] / confuse_EMO[:, i].sum(axis=0)
        infostr_EMO = dataset_info.info_EMO(torch.diag(confuse_EMO).cpu().numpy().tolist())
        logging.info(infostr_EMO)
        '''
        summary_writer.add_scalar('train_EMO_loss', train_EMO_loss, epoch)
        summary_writer.add_scalar('train_EMO_acc', train_EMO_acc, epoch)
        summary_writer.add_scalar('test_EMO_loss', test_EMO_loss, epoch)
        summary_writer.add_scalar('test_EMO_acc', test_EMO_acc, epoch)
        '''
        
        # RULESlog
        infostr_rules = {'Epoch {} val_rules_acc: {:.5f} val_rules_loss: {:.5f}'.format(epoch + 1, val_rules_acc, val_rules_loss)}
        logging.info(infostr_rules)
        '''
        summary_writer.add_scalar('train_rules_loss', train_rules_loss, epoch)
        summary_writer.add_scalar('train_rules_acc', train_rules_acc, epoch)
        summary_writer.add_scalar('val_rules_loss', val_rules_loss, epoch)
        summary_writer.add_scalar('val_rules_acc', val_rules_acc, epoch)
        '''
        #---------------------------------Logging Part--------------------------

        if (epoch+1) % conf.save_epoch == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict_AU': net_AU.state_dict(),
                'state_dict_EMO': net_EMO.state_dict(),
                'optimizer_AU': optimizer_AU.state_dict(),
                'optimizer_EMO': optimizer_EMO.state_dict(),
                'scheduler_EMO': scheduler_EMO.state_dict(),
                'rules': rules
            }
            torch.save(checkpoint, os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))
    checkpoint = {
            'epoch': epoch,
            'state_dict_AU': net_AU.state_dict(),
            'state_dict_EMO': net_EMO.state_dict(),
            'optimizer_AU': optimizer_AU.state_dict(),
            'optimizer_EMO': optimizer_EMO.state_dict(),
            'scheduler_EMO': scheduler_EMO.state_dict(),
            'rules': rules
        }
    torch.save(checkpoint, os.path.join(conf['outdir'], 'cur_model_fold' + str(conf.fold) + '.pth'))
    shutil.copyfile(os.path.join(conf['outdir'], 'train.log'), os.path.join(conf['outdir'], 'train_copy.log'))

if __name__=='__main__':
    conf = get_config()

    global device
    device = torch.device('cuda:{}'.format(conf.gpu))

    set_env(conf)
    set_outdir(conf) # generate outdir name
    set_logger(conf) # Set the logger
    main(conf)