import os
from re import A, M
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from conf import parser2dict, ensure_dir, get_config
from models.AU_EMO_BP import UpdateGraph_v2 as UpdateGraph
from models.RadiationAUs import RadiateAUs_v2 as RadiateAUs

from utils import *
from tensorboardX import SummaryWriter

import gc
import objgraph 

def randomPriori(input_rules):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules 
    EMO2AU_cpt = np.random.random(EMO2AU_cpt.shape)
    AU_cpt = np.zeros(AU_cpt.shape)
    for i, j in enumerate(AU[:-2]):
        prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
    ori_size = 0
    num_all_img = ori_size
    AU_ij_cnt = np.zeros(AU_ij_cnt.shape)
    AU_cnt = np.zeros_like(AU_cnt)
    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return output_rules

def learn_rules(conf, device, input_info, input_rules, summary_writer, AU_p_d, *args):
    lr_relation_flag = 0
    # init_lr = conf.lr_relation
    labelsAU, labelsEMO = input_info
    # input_rules = randomPriori(input_rules)
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    train_size = labelsAU.shape[0]
    init_lr = num_all_img / (num_all_img + train_size)

    if args:
        change_weight1 = num_all_img / (num_all_img + train_size)
        change_weight2 = 1
        for changing_item in args:
            change_weight2 = change_weight2 * changing_item
        init_lr = change_weight1 * change_weight2

    criterion = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    confu_m = torch.zeros((num_EMO, num_EMO))

    loc = list(range(EMO2AU_cpt.shape[1]))
    loc1 = loc[:-2]
    loc2 = loc[-2:]
    EMO2AU = EMO2AU_cpt

    update = UpdateGraph(conf, EMO2AU_cpt, prob_AU, loc1, loc2).to(device)
    # init_lr = 0.1
    optim_graph = optim.SGD(update.parameters(), lr=init_lr)
    update.train()

    for idx in range(labelsAU.shape[0]):
        torch.cuda.empty_cache()
        adjust_rules_lr_v2(optim_graph, init_lr, idx, train_size)
        cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
        emo_label = labelsEMO[idx].reshape(1,).to(device)

        occ_au = []
        prob_all_au = np.zeros((len(AU),))
        for i, au in enumerate(AU[:-2]):
            if cur_item[0, i] == 1:
                occ_au.append(i)
                AU_cnt[i] += 1

        if cur_item.sum() != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU, thresh=0.6) # 计算当前样本中AU的发生概率 P(AU | x)
            cur_prob = update(prob_all_au)
            cur_pred = torch.argmax(cur_prob)
            optim_graph.zero_grad()
            err = criterion(cur_prob, emo_label)
            acc = torch.eq(cur_pred, emo_label).sum().item()
            err_record.append(err.item())
            acc_record.append(acc)
            confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
            summary_writer.add_scalar('train_err', np.array(err_record).mean(), idx)
            summary_writer.add_scalar('train_acc', np.array(acc_record).mean(), idx)
            
            err.backward()
            optim_graph.step()

            EMO2AU_cpt = update.EMO2AU_cpt.data.detach().cpu().numpy()
            prob_AU = update.prob_AU.data.detach().cpu().numpy()
            EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
            EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)
            update.EMO2AU_cpt.data.copy_(torch.from_numpy(EMO2AU_cpt))
            for i, au_i in enumerate(occ_au):
                for j, au_j in enumerate(occ_au):
                    if i != j:
                        AU_ij_cnt[au_i][au_j] = AU_ij_cnt[au_i][au_j]+1
                        AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
            for i, j in enumerate(AU[:-2]):
                prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            update.prob_AU.data.copy_(torch.from_numpy(prob_AU))

            EMO2AU_cpt1 = update.EMO2AU_cpt.data
            EMO2AU_cpt2 = update.static_EMO2AU_cpt.data
            EMO2AU = torch.cat((EMO2AU_cpt1, EMO2AU_cpt2), dim=1).detach().cpu().numpy()

        if idx >= 20000:
            break
    
    EMO2AU_cpt1 = update.EMO2AU_cpt.data
    EMO2AU_cpt2 = update.static_EMO2AU_cpt.data
    EMO2AU_cpt = torch.cat((EMO2AU_cpt1, EMO2AU_cpt2), dim=1).detach().cpu().numpy()
    prob_AU1 = update.prob_AU.data
    prob_AU2 = update.static_prob_AU.data
    prob_AU = torch.cat((prob_AU1, prob_AU2)).detach().cpu().numpy()

    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return output_rules, output_records, update

def test_rules(conf, update, device, input_info, input_rules, AU_p_d, summary_writer):

    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    
    criterion = nn.CrossEntropyLoss()
    acc_record = []
    err_record = []
    num_EMO = EMO2AU_cpt.shape[0]
    confu_m = torch.zeros((num_EMO, num_EMO))

    loc = list(range(EMO2AU_cpt.shape[1]))
    loc1 = loc[:-2]
    loc2 = loc[-2:]

    # update = UpdateGraph_v2(conf, EMO2AU_cpt, prob_AU, loc1, loc2).to(device)
    update = update.to(device)
    update.eval()

    with torch.no_grad():
        for idx in range(labelsAU.shape[0]):
            torch.cuda.empty_cache()
            cur_item = labelsAU[idx, :].reshape(1, -1).to(device)
            emo_label = labelsEMO[idx].reshape(1,).to(device)

            occ_au = []
            prob_all_au = np.zeros((len(AU),))
            for i, au in enumerate(AU[:-2]):
                if cur_item[0, i] == 1:
                    occ_au.append(i)

            if cur_item.sum() != 0:
                prob_all_au = RadiateAUs(conf, emo_label, AU_cpt, occ_au, loc2, EMO2AU_cpt, thresh=0.6)
                cur_prob = update(prob_all_au)
                cur_pred = torch.argmax(cur_prob)
                confu_m = confusion_matrix(cur_pred.data.cpu().numpy().reshape(1,).tolist(), labels=emo_label.data.cpu().numpy().tolist(), conf_matrix=confu_m)
                err = criterion(cur_prob, emo_label)
                acc = torch.eq(cur_pred, emo_label).sum().item()
                err_record.append(err.item())
                acc_record.append(acc)
                summary_writer.add_scalar('val_err', np.array(err_record).mean(), idx)
                summary_writer.add_scalar('val_acc', np.array(acc_record).mean(), idx)
    if len(err_record) == 0:
        output_records = (0, 0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean(), confu_m)
    return output_records

def main_rules(conf, device, cur_path, info_source, AU_p_d):
    pre_path = conf.outdir
    info_path = os.path.join(pre_path, cur_path)

    info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
    # rules_summary_path = os.path.join(pre_path, 'Priori', info_source_path, cur_path.split('.')[0])
    rules_summary_path = '/media/data1/wf/AU_EMOwPGM/codes/results/tmp'
    ensure_dir(rules_summary_path, 0)
    summary_writer = SummaryWriter(rules_summary_path)
    
    all_info = torch.load(info_path, map_location='cpu')#['state_dict']
    input_rules = all_info['input_rules']
    train_rules_input = (all_info['train_input_info'][info_source[0]], all_info['train_input_info'][info_source[1]])
    val_rules_input = (all_info['val_input_info'][info_source[0]], all_info['val_input_info'][info_source[1]])

    change_w = 0.1
    output_rules, train_records, model = learn_rules(conf, device, train_rules_input, input_rules, summary_writer, AU_p_d, change_w)
    train_rules_loss, train_rules_acc, train_confu_m = train_records
    val_records = test_rules(conf, model, device, val_rules_input, output_rules, AU_p_d, summary_writer)
    # val_records = test_rules(conf, device, val_rules_input, input_rules, AU_p_d, summary_writer)
    val_rules_loss, val_rules_acc, val_confu_m = val_records

    checkpoint = {}
    checkpoint['input_rules'] = input_rules
    checkpoint['output_rules'] = output_rules
    checkpoint['train_records'] = train_records
    checkpoint['val_records'] = val_records
    checkpoint['model'] = model
    # torch.save(checkpoint, os.path.join(pre_path, info_source_path, cur_path))
    torch.save(checkpoint, os.path.join(rules_summary_path, 'tmp.pth'))
    end_flag = True

    return model, train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc, val_confu_m

def main_cross(conf, device, cur_path, info_source, AU_p_d):
    pre_path = conf.outdir
    info_path = os.path.join(pre_path, cur_path)

    info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
    rules_summary_path = os.path.join(pre_path, info_source_path, 'usingBP4D', cur_path.split('.')[0])
    # rules_summary_path = '/media/data1/wf/AU_EMOwPGM/codes/results/tmp'
    ensure_dir(rules_summary_path, 0)
    summary_writer = SummaryWriter(rules_summary_path)

    rules_path = '/media/data1/wf/AU_EMOwPGM/codes/results/BP4D/Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/labelsAU_labelsEMO/epoch4_model_fold0.pth'
    output_rules = torch.load(rules_path, map_location='cpu')['output_rules']
    # model = torch.load(rules_path, map_location='cpu')['model']

    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = output_rules
    loc = list(range(EMO2AU_cpt.shape[1]))
    loc1 = loc[:-2]
    loc2 = loc[-2:]
    EMO2AU = EMO2AU_cpt
    model = UpdateGraph(conf, EMO2AU_cpt, prob_AU, loc1, loc2).to(device)

    all_info = torch.load(info_path, map_location='cpu')
    val_rules_input = (all_info['val_input_info'][info_source[0]], all_info['val_input_info'][info_source[1]])
    val_records = test_rules(conf, model, device, val_rules_input, output_rules, AU_p_d, summary_writer)
    val_rules_loss, val_rules_acc, val_confu_m = val_records

    checkpoint = {}
    checkpoint['val_records'] = val_records
    torch.save(checkpoint, os.path.join(rules_summary_path, 'out.pth'))

    return val_rules_loss, val_rules_acc

def main_continuous(conf, device, cur_path, info_source, AU_p_d):
    pre_path = conf.outdir
    info_path = os.path.join(pre_path, cur_path)

    info_source_path = info_source[0].split('_')[0] + '_' + info_source[1].split('_')[0]
    # rules_summary_path = os.path.join(pre_path, 'BP4D_continuous', info_source_path, cur_path.split('.')[0])
    rules_summary_path = '/media/data1/wf/AU_EMOwPGM/codes/results/tmp'
    ensure_dir(rules_summary_path, 0)
    summary_writer = SummaryWriter(rules_summary_path)

    rules_path = '/media/data1/wf/AU_EMOwPGM/codes/results0911_theTrust/BP4D/Test/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/labelsAU_labelsEMO/epoch8_model_fold0.pth'
    input_rules = torch.load(rules_path, map_location='cpu')['output_rules']
    all_info = torch.load(info_path, map_location='cpu')#['state_dict']

    train_rules_input = (all_info['train_input_info'][info_source[0]], all_info['train_input_info'][info_source[1]])
    val_rules_input = (all_info['val_input_info'][info_source[0]], all_info['val_input_info'][info_source[1]])
    conf.lr_relation = 0.005
    # conf.lr_relation = 0.001
    # conf.lr_decay_idx = 2000

    output_rules, output_records = learn_rules(conf, device, train_rules_input, input_rules, summary_writer, AU_p_d)
    train_rules_loss, train_rules_acc, train_confu_m = output_records
    output_records = test_rules(conf, device, val_rules_input, output_rules, summary_writer, AU_p_d)
    val_rules_loss, val_rules_acc, val_confu_m = output_records

    rules_path_all_info = torch.load('/media/data1/wf/AU_EMOwPGM/codes/results0911_theTrust/BP4D/Test/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/'+cur_path, map_location='cpu')
    BP4D_val_input = (rules_path_all_info ['val_input_info']['labelsAU_record'], rules_path_all_info ['val_input_info']['labelsEMO_record'])
    # temp_summary_path = os.path.join(pre_path, 'BP4D_continuous', 'source', info_source_path, cur_path.split('.')[0])
    temp_summary_path = '/media/data1/wf/AU_EMOwPGM/codes/results/tmp2'
    ensure_dir(temp_summary_path, 0)
    temp_summary_writer = SummaryWriter(temp_summary_path)
    BP4D_records = test_rules(conf, device, BP4D_val_input, output_rules, temp_summary_writer, AU_p_d)
    BP4D_rules_loss, BP4D_rules_acc, BP4D_confu_m = BP4D_records

    all_info['output_rules'] = output_rules
    torch.save(all_info, info_path)

    return train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc

def main(conf):
    train_loader, test_loader, train_len, test_len = getDatasetInfo(conf)
    dataset_AU = train_loader.dataset.AU
    priori_AU = train_loader.dataset.priori['AU']
    AU_p_d = (priori_AU, dataset_AU)
    source_list = [
        ['predsAU_record', 'labelsEMO_record'],
        # ['labelsAU_record', 'predsEMO_record'],
        # ['labelsAU_record', 'labelsEMO_record'],
        # ['predsAU_record', 'predsEMO_record']
        ]
    pre_path = '/media/data1/wf/AU_EMOwPGM/codes/results/AffectNet/Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    conf.outdir = pre_path
    # file_list = walkFile(pre_path)
    file_list = ['epoch1_model_fold0.pth']
    
    for info_source in source_list:
        for f in file_list:
            torch.cuda.empty_cache()
            print('The current info source are %s and %s, the features are from %s '%(info_source[0], info_source[1], f))
            model, train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc, val_confu_m = main_rules(conf, device, f, info_source, AU_p_d)
            print('train_rules_loss: {:.5f}, train_rules_acc: {:.5f}, val_rules_loss: {:.5f},, val_rules_acc: {:.5f},'
                                    .format(train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc))
            del train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc
            # val_rules_loss, val_rules_acc = main_cross(conf, device, f, info_source, AU_p_d)
            # print('val_rules_loss: {:.5f}, val_rules_acc: {:.5f},' .format(val_rules_loss, val_rules_acc))
            # del val_rules_loss, val_rules_acc

if __name__=='__main__':
    conf = parser2dict()
    conf.dataset = 'AffectNet'
    conf = get_config(conf)
    conf.gpu = 1

    global device
    device = torch.device('cuda:{}'.format(conf.gpu))
    conf.device = device
    torch.cuda.set_device(conf.gpu)
    main(conf)
    a = 1

# def train(model,epochs,data):
# for e in range(epochs):
#     print("1:{}".format(torch.cuda.memory_allocated(0)))
#     train_epoch(model,data)
#     print("2:{}".format(torch.cuda.memory_allocated(0)))
#     eval(model,data)
#     print("3:{}".format(torch.cuda.memory_allocated(0)))