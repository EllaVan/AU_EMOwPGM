import os
# os.chdir(os.path.dirname(__file__))
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.AU_EMO_BP import UpdateGraph
from models.RadiationAUs import RadiateAUs
# from AU_EMO_BP import UpdateGraph
# from RadiationAUs import RadiateAUs
# import rules_learning.utils as utils
# from tensorboardX import SummaryWriter
# from conf import get_config

def learn_rules(conf, input_info, input_rules, lr):
    lr_relation_flag = 0
    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    labelsAU, labelsEMO = labelsAU.to(conf.device), labelsEMO.to(conf.device)

    AU_evidence = torch.ones((1, 1)).to(conf.device)
    output_records = (0, 0)
    acc_record = []
    err_record = []
    for idx in range(labelsAU.shape[0]):
        cur_item = labelsAU[idx, :].reshape(1, -1)
        emo_label = labelsEMO[idx].reshape(1,)
        torch.cuda.empty_cache()
        weight = []
        occ_au = []
        prob_all_au = np.zeros((len(AU),))
        for i, au in enumerate(AU[:-2]):
            if cur_item[0, i] == 1:
                occ_au.append(i)
                prob_all_au[i] = 1
                AU_cnt[i] += 1
            weight.append(EMO2AU_cpt[:, i])
        weight.append(EMO2AU_cpt[:, -2])
        weight.append(EMO2AU_cpt[:, -1])

        if len(occ_au) != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
            pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
            weight = np.array(weight)
            for i in range(prob_all_au.shape[0]-2):
                if i in pos:
                    prob_all_au[i] = prob_all_au[i] / prob_AU[i]
                else:
                    prob_all_au[i] = 1 / (1-prob_AU[i])
                    weight[i, :] = 1 - weight[i, :]
            prob_all_au[-2] = 1 / prob_AU[-2]
            prob_all_au[-1] = 1 / prob_AU[-1]
            if emo_label == 0:
                weight[-1, :] = 1 - weight[-1, :]
                prob_all_au[-1] = 1 / (1-prob_AU[-1])
            elif emo_label == 2 or emo_label == 4:
                pass
            else:
                weight[-2, :] = 1 - weight[-2, :]
                weight[-1, :] = 1 - weight[-1, :]
                prob_all_au[-2] = 1 / (1-prob_AU[-2])
                prob_all_au[-1] = 1 / (1-prob_AU[-1])
            init = np.ones((1, len(EMO)))
            for i in range(weight.shape[1]):
                for j in range(1, 3):
                    init[:, i] *= weight[-j][i]*prob_all_au[-j]
            
            weight = np.where(weight > 0, weight, conf.zeroPad)
            update = UpdateGraph(conf, in_channels=1, out_channels=len(EMO), W=weight[:-2, :], 
                                prob_all_au=prob_all_au[:-2], init=init).to(conf.device)
            optim_graph = optim.SGD(update.parameters(), lr=lr)
            
            cur_prob = update(AU_evidence)
            cur_pred = torch.argmax(cur_prob)
            err = nn.CrossEntropyLoss()(cur_prob, emo_label)
            acc = torch.eq(cur_pred, emo_label).type(torch.FloatTensor).item()
            err_record.append(err.item())
            acc_record.append(acc)

            optim_graph.zero_grad()
            err.backward()
            optim_graph.step()
            
            update_info1 = update.fc.weight.grad.cpu().numpy().squeeze()
            update_info2 = update.d1.detach().cpu().numpy().squeeze()
            for emo_i, emo_name in enumerate(EMO):
                for i, j in enumerate(AU[:-2]):
                    factor = update_info2[emo_i] / weight[i, emo_i]
                    weight[i, emo_i] -= lr*update_info1[emo_i]*factor
                    if i in pos:
                        EMO2AU_cpt[emo_i, i] = weight[i, emo_i]
                    else:
                        EMO2AU_cpt[emo_i, i] = 1-weight[i, emo_i]
            EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
            EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)

            for i, au_i in enumerate(occ_au):
                for j, au_j in enumerate(occ_au):
                    if i != j:
                        AU_ij_cnt[au_i][au_j] += 1
                        AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
            for i, j in enumerate(AU[:-2]):
                prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (len(EMO))
            prob_AU = np.where(prob_AU > 0, prob_AU, conf.zeroPad)
            prob_AU = np.where(prob_AU <= 1, prob_AU, 1)

        if num_all_img-ori_size >= conf.lr_decay_idx and lr_relation_flag == 0:
            lr_relation_flag = 1
            lr /= 10.0

    if len(err_record) == 0:
        output_records = (0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean())
    output_rules = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return output_rules, output_records

def test_rules(conf, input_info, input_rules):

    labelsAU, labelsEMO = input_info
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = input_rules
    labelsAU, labelsEMO = labelsAU.to(conf.device), labelsEMO.to(conf.device)
    
    AU_evidence = torch.ones((1, 1)).to(conf.device)
    acc_record = []
    err_record = []

    for idx in range(labelsAU.shape[0]):
        cur_item = labelsAU[idx, :].reshape(1, -1)
        emo_label = labelsEMO[idx].reshape(1,)
        torch.cuda.empty_cache()
        weight = []
        occ_au = []
        prob_all_au = np.zeros((len(AU),))
        for i, au in enumerate(AU[:-2]):
            if cur_item[0, i] == 1:
                occ_au.append(i)
                prob_all_au[i] = 1
            weight.append(EMO2AU_cpt[:, i])
        weight.append(EMO2AU_cpt[:, -2])
        weight.append(EMO2AU_cpt[:, -1])

        if len(occ_au) != 0:
            num_all_img += 1
            prob_all_au = RadiateAUs(AU_cpt, occ_au, thresh=0.6)
            pos = np.where(prob_all_au > 0.6)[0] # pos = np.where(prob_all_au == 1)[0]
            weight = np.array(weight)
            for i in range(prob_all_au.shape[0]-2):
                if i in pos:
                    prob_all_au[i] = prob_all_au[i] / prob_AU[i]
                else:
                    prob_all_au[i] = 1 / (1-prob_AU[i])
                    weight[i, :] = 1 - weight[i, :]
            prob_all_au[-2] = 1 / prob_AU[-2]
            prob_all_au[-1] = 1 / prob_AU[-1]
            if emo_label == 0:
                weight[-1, :] = 1 - weight[-1, :]
                prob_all_au[-1] = 1 / (1-prob_AU[-1])
            elif emo_label == 2 or emo_label == 4:
                pass
            else:
                weight[-2, :] = 1 - weight[-2, :]
                weight[-1, :] = 1 - weight[-1, :]
                prob_all_au[-2] = 1 / (1-prob_AU[-2])
                prob_all_au[-1] = 1 / (1-prob_AU[-1])
            init = np.ones((1, len(EMO)))
            for i in range(weight.shape[1]):
                for j in range(1, 3):
                    init[:, i] *= weight[-j][i]*prob_all_au[-j]
            
            weight = np.where(weight > 0, weight, conf.zeroPad)
            update = UpdateGraph(conf, in_channels=1, out_channels=len(EMO), W=weight[:-2, :], 
                                prob_all_au=prob_all_au[:-2], init=init).to(conf.device)
            
            cur_prob = update(AU_evidence)
            cur_pred = torch.argmax(cur_prob)
            err = nn.CrossEntropyLoss()(cur_prob, emo_label)
            acc = torch.eq(cur_pred, emo_label).type(torch.FloatTensor).item()
            err_record.append(err.item())
            acc_record.append(acc)
    if len(err_record) == 0:
        output_records = (0, 0)
    else:
        output_records = (np.array(err_record).mean(), np.array(acc_record).mean())
    return output_records


def main(conf):
    pre_path = '/media/data1/wf/AU_EMOwPGM/codes/results/Test/bs_64_seed_0_lrEMO_0.0004_lrAU_0.0001_lr_relation_0.001_tmp'
    info_path = os.path.join(pre_path, 'epoch4_model_fold1.pth')
    save_path = pre_path
    all_info = torch.load(info_path, map_location='cpu')#['state_dict']
    input_rules = all_info['input_rules']
    train_input_info = all_info['train_input_info']
    val_input_info = all_info['val_input_info']

    output_rules, output_records = learn_rules(conf, train_input_info, input_rules, conf.lr_relation, save_path)
    train_rules_loss, train_rules_acc = output_records
    output_records = test_rules(conf, val_input_info, output_rules, save_path)
    val_rules_loss, val_rules_acc = output_records

    print('train_rules_loss: {:.5f}, train_rules_acc: {:.5f},, val_rules_loss: {:.5f},, val_rules_acc: {:.5f},'
            .format(train_rules_loss, train_rules_acc, val_rules_loss, val_rules_acc))

# if __name__=='__main__':
    
#     conf = get_config()
    # global device
    # device = torch.device('cuda:{}'.format(2))
    # conf.device = device
    # main(conf)