from easydict import EasyDict as edict
import yaml
import random

import torch
import numpy as np

def read_file():
    path1 = 'extending/save/unseen/ZSL/2022-10-30/BP4D/output.pth'
    file1 = torch.load(path1, map_location='cpu')
    extend_EMO2AU = file1['output_rules'][0]
    path2 = '/media/data1/wf/AU_EMOwPGM/codes/results/save/BP4D/labelsAU_labelsEMO/epoch4_model_fold0.pth'
    file2 = torch.load(path2, map_location='cpu')
    a = file2['output_rules'][0][[0, 1, 3, 4], :]
    b = file2['output_rules'][0][[2, 5], :]
    trained_EMO2AU = np.concatenate([a, b])
    a = 1

def get_config(cfg):
    if cfg.dataset == 'BP4D':
        with open('../config/BP4D_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'BP4D_all':
        with open('../config/BP4D_all_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch1_model_fold0.pth'
    elif cfg.dataset == 'DISFA':
        with open('../config/DISFA_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'predsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'RAF-DB':
        with open('../config/RAF_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch4_model_fold0.pth'
    elif cfg.dataset == 'RAF-DB-compound':
        with open('../config/RAF_compound_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
    elif cfg.dataset == 'AffectNet':
        with open('../config/AffectNet_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['predsAU_record', 'labelsEMO_record']
            cfg.file_list = 'epoch1_model_fold0.pth'
    elif cfg.dataset == 'CASME':
        with open('../config/CASME_config.yaml', 'r') as f:
            datasets_cfg = yaml.safe_load(f)
            datasets_cfg = edict(datasets_cfg)
            cfg.source_list = ['labelsAU_record', 'labelsEMO_record']
    else:
        raise Exception("Unkown Datsets:",cfg.dataset)

    if cfg.fold == 0:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_independent
    else:
        datasets_cfg.dataset_path = datasets_cfg.dataset_path_subject_dependent
    cfg.update(datasets_cfg)
    return cfg

def generate_seen_sample(conf, rules, topk=8):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = rules
    loc1 = conf.loc1
    loc2 = conf.loc2
    refer_loc = list(range(len(loc1)))
    num_occu_AU = len(loc1)
    samplesEMO = []
    samplesAU = []
    for emo_i, emo in enumerate(EMO):
        sampleEMO = torch.from_numpy(np.array(emo_i).reshape(1, ))
        samplesEMO.append(sampleEMO)
        samplesEMO.append(sampleEMO)
        
        least_sampleAU = torch.zeros((1, num_occu_AU))
        next_least_sampleAU = torch.zeros((1, num_occu_AU))

        sort_EMO2AU = EMO2AU_cpt[emo_i, loc1].copy()
        cur_EMO2AU = np.argsort(-sort_EMO2AU)
        topk_idx = cur_EMO2AU[:topk]
        half_topk_idx = cur_EMO2AU[:int(topk/2)]
        lowk = len(loc1) - topk
        lowk_idx = cur_EMO2AU[-lowk:]
        half_lowk_idx = cur_EMO2AU[-lowk:-int(lowk/2)]

        least_sampleAU[:, topk_idx] = 1
        least_sampleAU[:, half_lowk_idx] = 1
        next_least_sampleAU[:, half_topk_idx] = 1
        next_least_sampleAU[:, lowk_idx] = 1
        samplesAU.append(least_sampleAU)
        samplesAU.append(next_least_sampleAU)

        list_topk_idx = list(topk_idx)
        list_lowk_idx = list(lowk_idx)
        for t_sample in range(1, 5):
            sample_AU_t = torch.zeros((1, num_occu_AU))
            top_key_size = random.randint(3, topk+1)
            low_key_size = random.randint(1, lowk+1)
            top_key1 = random.sample(list_topk_idx, top_key_size-1)
            sample_AU_t[:, top_key1] = 1
            low_key1 = random.sample(list_lowk_idx, low_key_size-1)
            sample_AU_t[:, low_key1] = 1
            samplesEMO.append(sampleEMO)
            samplesAU.append(sample_AU_t)

    # samplesEMO = torch.concat(samplesEMO)
    samplesAU = torch.concat(samplesAU)
    return samplesAU, samplesEMO

def generate_seen_sample_v2(conf, rules, topk=8):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = rules
    AU2EMO_cpt = np.zeros(EMO2AU_cpt.T.shape)
    for i, au in enumerate(AU):
        for j, emo in enumerate(EMO):
            AU2EMO_cpt[i, j] = EMO2AU_cpt[j, i] / len(EMO) / prob_AU[i] # 注意此处的 /len(EMO) 表达的其实是p(EMO)，此处存在各个类别均匀分布的假设

    loc1 = conf.loc1
    loc2 = conf.loc2
    refer_loc = list(range(len(loc1)))
    num_occu_AU = len(loc1)
    samplesEMO = []
    samplesAU = []
    for emo_i, emo in enumerate(EMO):
        sampleEMO = torch.from_numpy(np.array(emo_i).reshape(1, ))
        samplesEMO.append(sampleEMO)
        samplesEMO.append(sampleEMO)
        
        least_sampleAU = torch.zeros((1, num_occu_AU))
        next_least_sampleAU = torch.zeros((1, num_occu_AU))

        sort_AU2EMO = AU2EMO_cpt[loc1, emo_i].copy()
        cur_AU2EMO = np.argsort(-sort_AU2EMO)
        topk_idx = cur_AU2EMO[:topk]
        half_topk_idx = cur_AU2EMO[:int(topk/2)]
        lowk = len(loc1) - topk
        lowk_idx = cur_AU2EMO[-lowk:]
        half_lowk_idx = cur_AU2EMO[-lowk:-int(lowk/2)]

        least_sampleAU[:, topk_idx] = 1
        least_sampleAU[:, half_lowk_idx] = 1
        next_least_sampleAU[:, half_topk_idx] = 1
        next_least_sampleAU[:, lowk_idx] = 1
        samplesAU.append(least_sampleAU)
        samplesAU.append(next_least_sampleAU)

        list_topk_idx = list(topk_idx)
        list_lowk_idx = list(lowk_idx)
        for t_sample in range(1, 5):
            sample_AU_t = torch.zeros((1, num_occu_AU))
            top_key_size = random.randint(3, topk+1)
            low_key_size = random.randint(1, lowk+1)
            top_key1 = random.sample(list_topk_idx, top_key_size-1)
            sample_AU_t[:, top_key1] = 1
            low_key1 = random.sample(list_lowk_idx, low_key_size-1)
            sample_AU_t[:, low_key1] = 1
            samplesEMO.append(sampleEMO)
            samplesAU.append(sample_AU_t)

    # samplesEMO = torch.concat(samplesEMO)
    samplesAU = torch.concat(samplesAU)
    return samplesAU, samplesEMO        


def sample_seen(EMO, train_inputAU, train_inputEMO, ori_samplek):
    samplesEMO = []
    samplesAU = []
    for emo_i, emo in enumerate(EMO):
        emoi_loc = torch.where(train_inputEMO==emo_i)[0].numpy().tolist()
        if len(emoi_loc) < ori_samplek:
            samplek = len(emoi_loc)-1
        else:
            samplek = ori_samplek
        sample_loc = random.sample(emoi_loc, k=samplek)
        samplesEMO.append(train_inputEMO[sample_loc])
        samplesAU.append(train_inputAU[sample_loc, :])
    samplesAU = torch.concat(samplesAU)
    samplesEMO = torch.concat(samplesEMO)
    return samplesAU, samplesEMO

# 按照unseen_priori的EMO2AU情况，得到unseen_priori_rule
def get_unseen_priori_rule(train_loader, unseen_loc=[2,5]):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())
    unseen_EMO2AU_cpt = EMO2AU_cpt[unseen_loc, :]

    num_AU = len(AU)
    num_unseen = len(unseen_loc)
    AU_cpt_tmp = np.zeros((num_AU, num_AU))  #初始化AU的联合概率表
    for k in range(num_unseen):
        AU_pos_nonzero = np.nonzero(unseen_EMO2AU_cpt[k])[0]  #EMO=k的AU-EMO关联（非0值）的位置
        AU_pos_certain = np.where(unseen_EMO2AU_cpt[k]==1)[0]  #EMO=k的AU-EMO关联（1值）的位置
        for j in range(len(AU_pos_nonzero)):
            if unseen_EMO2AU_cpt[k][AU_pos_nonzero[j]] == 1.0:  #当AUj是确定发生的时候，AUi同时发生的概率就是AUi自己本身的值
                for i in range(len(AU_pos_nonzero)):
                    if i != j:
                        AU_cpt_tmp[AU_pos_nonzero[i]][AU_pos_nonzero[j]] += unseen_EMO2AU_cpt[k][AU_pos_nonzero[i]]
            else:  #而当AUj的发生是不确定的时候，只初始化确定发生的AUi，值为1
                for i in range(len(AU_pos_certain)):
                    AU_cpt_tmp[AU_pos_certain[i]][AU_pos_nonzero[j]] += 1.0
    unseen_AU_cpt = (AU_cpt_tmp / num_unseen) + np.eye(num_AU)
    unseen_prob_AU = np.sum(unseen_EMO2AU_cpt, axis=0) / num_unseen
    EMO_img_num = [230] * num_unseen
    unseen_ori_size = np.sum(np.array(EMO_img_num))
    unseen_num_all_img = unseen_ori_size
    unseen_AU_cnt = unseen_prob_AU * unseen_ori_size
    unseen_AU_ij_cnt = np.zeros_like(unseen_AU_cpt)
    for au_ij in range(unseen_AU_cpt.shape[0]):
        unseen_AU_ij_cnt[:, au_ij] = unseen_AU_cpt[:, au_ij] * unseen_AU_cnt[au_ij]
    EMO = ['fear', 'disgust']
    unseen_rule = unseen_EMO2AU_cpt, unseen_AU_cpt, unseen_prob_AU, unseen_ori_size, unseen_num_all_img, unseen_AU_ij_cnt, unseen_AU_cnt, EMO, AU

    return unseen_rule

# 根据(seen, unseen)得到整体的rule
def get_complete_rule(seen_rules, unseen_rules):
    seen_EMO2AU_cpt, seen_AU_cpt, seen_prob_AU, seen_ori_size, seen_num_all_img, seen_AU_ij_cnt, seen_AU_cnt, seen_EMO, AU = seen_rules
    unseen_EMO2AU_cpt, unseen_AU_cpt, unseen_prob_AU, unseen_ori_size, unseen_num_all_img, unseen_AU_ij_cnt, unseen_AU_cnt, unseen_EMO, AU = unseen_rules
    EMO2AU_cpt = np.concatenate((seen_EMO2AU_cpt, unseen_EMO2AU_cpt))
    EMO = seen_EMO + unseen_EMO

    num_AU = len(AU)
    num_seen = seen_EMO2AU_cpt.shape[0]
    num_unseen = unseen_EMO2AU_cpt.shape[0]
    num_EMO = num_seen + num_unseen

    # prob_AU = num_unseen/(num_unseen+num_seen)*unseen_prob_AU + num_seen/(num_unseen+num_seen)*seen_prob_AU
    prob_AU = np.sum(EMO2AU_cpt, axis=0) / num_EMO
    ori_size = seen_ori_size + unseen_ori_size
    num_all_img = seen_num_all_img + unseen_num_all_img

    AU_ij_cnt = seen_AU_ij_cnt + unseen_AU_ij_cnt
    AU_cnt = seen_AU_cnt + unseen_AU_cnt
    AU_cpt = np.zeros((num_AU, num_AU)) + np.eye(num_AU)
    for au_i in range(num_AU):
        for au_j in range(num_AU):
            if au_i != au_j:
                AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]

    # AU_cnt = prob_AU * num_all_img
    # AU_cpt = num_unseen/(num_unseen+num_seen)*unseen_AU_cpt + num_seen/(num_unseen+num_seen)*seen_AU_cpt

    complete_rule = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    # complete_rule = EMO2AU_cpt, seen_AU_cpt, seen_prob_AU, seen_ori_size, seen_num_all_img, seen_AU_ij_cnt, seen_AU_cnt, EMO, AU
    return complete_rule

def get_cat_rule(seen_rules, unseen_rules):
    seen_EMO2AU_cpt, seen_AU_cpt, seen_prob_AU, seen_ori_size, seen_num_all_img, seen_AU_ij_cnt, seen_AU_cnt, seen_EMO, AU = seen_rules
    unseen_EMO2AU_cpt, unseen_AU_cpt, unseen_prob_AU, unseen_ori_size, unseen_num_all_img, unseen_AU_ij_cnt, unseen_AU_cnt, unseen_EMO, AU = unseen_rules
    EMO2AU_cpt = np.concatenate((seen_EMO2AU_cpt, unseen_EMO2AU_cpt))
    EMO = seen_EMO + unseen_EMO

    num_AU = len(AU)
    num_seen = seen_EMO2AU_cpt.shape[0]
    num_unseen = unseen_EMO2AU_cpt.shape[0]
    num_EMO = num_seen + num_unseen

    prob_AU = np.sum(EMO2AU_cpt, axis=0) / num_EMO
    ori_size = seen_ori_size + unseen_ori_size
    num_all_img = seen_num_all_img + unseen_num_all_img

    AU_ij_cnt = seen_AU_ij_cnt + unseen_AU_ij_cnt
    AU_cnt = seen_AU_cnt + unseen_AU_cnt
    AU_cpt = np.zeros((num_AU, num_AU)) + np.eye(num_AU)
    for au_i in range(num_AU):
        for au_j in range(num_AU):
            if au_i != au_j:
                AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]

    complete_rule = EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU
    return complete_rule

def rule_order():
    order_name = ['EMO2AU_cpt', 'AU_cpt', 'prob_AU', 'ori_size', 'num_all_img', 'AU_ij_cnt', 'AU_cnt', 'EMO', 'AU']
    order_num = list(range(len(order_name)))
    rule_order = dict(zip(order_name, order_num))
    return rule_order

def crop_EMO2AU(conf, priori_update, *args):
    EMO2AU_cpt = priori_update.EMO2AU_cpt.data.detach().cpu().numpy()
    prob_AU = priori_update.prob_AU.data.detach().cpu().numpy()
    EMO2AU_cpt = np.where(EMO2AU_cpt > 0, EMO2AU_cpt, conf.zeroPad)
    EMO2AU_cpt = np.where(EMO2AU_cpt <= 1, EMO2AU_cpt, 1)
    priori_update.EMO2AU_cpt.data.copy_(torch.from_numpy(EMO2AU_cpt))
    loc1 = priori_update.loc1
    loc2 = priori_update.loc2
    for i, j in enumerate(loc1):
        prob_AU[i] = np.sum(EMO2AU_cpt[:, i]) / (EMO2AU_cpt.shape[0])
    priori_update.prob_AU.data.copy_(torch.from_numpy(prob_AU))
    if len(args) != 0:
        occ_au, AU_ij_cnt, AU_cpt, AU_cnt = args
        for i, au_i in enumerate(occ_au):
            for j, au_j in enumerate(occ_au):
                if i != j:
                    AU_ij_cnt[au_i][au_j] = AU_ij_cnt[au_i][au_j]+1
                    AU_cpt[au_i][au_j] = AU_ij_cnt[au_i][au_j] / AU_cnt[au_j]
        return priori_update, AU_ij_cnt, AU_cpt, AU_cnt
    else:
        return priori_update

def final_return(priori_update, EMO, AU, loc1, loc2):
    # EMO2AU_cpt = np.zeros((len(EMO), len(AU)))
    EMO2AU_cpt1 = priori_update.EMO2AU_cpt.data.detach().cpu().numpy()
    EMO2AU_cpt2 = priori_update.static_EMO2AU_cpt.data.detach().cpu().numpy()
    EMO2AU_cpt = np.zeros((EMO2AU_cpt1.shape[0], len(AU)))
    EMO2AU_cpt[:, loc1] = EMO2AU_cpt1
    EMO2AU_cpt[:, loc2] = EMO2AU_cpt2
    prob_AU = np.zeros((len(AU),))
    prob_AU1 = priori_update.prob_AU.data.detach().cpu().numpy()
    prob_AU2 = priori_update.static_prob_AU.data.detach().cpu().numpy()
    prob_AU[loc1] = prob_AU1
    prob_AU[loc2] = prob_AU2
    return priori_update, EMO2AU_cpt, prob_AU

def shuffle_input(inputAU, inputEMO):
    a = list(zip(inputAU, inputEMO))
    random.shuffle(a) 
    b = [x[0] for x in a]
    c = [x[1] for x in a]
    inputAU = torch.stack(b)
    inputEMO = torch.stack(c)
    return inputAU, inputEMO
