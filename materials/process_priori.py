import os

import pandas as pd
import numpy as np


def cal_interAUPriori():
    cur_path = os.path.dirname(__file__)
    EMO2AU_df = pd.read_csv(os.path.join(cur_path, 'AU_EMO_priori.csv'))
    EMO2AU_df = EMO2AU_df.fillna(0)
    
    EMO = list(EMO2AU_df.iloc[:, 0])
    AU = list(EMO2AU_df.columns.values[1:])

    EMO2AU_np = np.array(EMO2AU_df.iloc[:, 1:])
    num_EMO = EMO2AU_np.shape[0]
    num_AU = EMO2AU_np.shape[1]
    prob_AU = np.sum(EMO2AU_np, axis=0) / num_EMO  #AU的边缘概率
    EMO_img_num = [230] * 6  #各个EMO类别中图片的数量

    # 计算初始化的AU条件概率，AU_jpt[i][j] = P(AUi | AUj)
    # 初始化未考虑当一个表情中，两种AU均非确定性发生的情况(即对于一个确定的表情而言，AUi和AUj必有一个条件概率为1.0)
    AU_cpt_tmp = np.zeros((num_AU, num_AU))  #初始化AU的联合概率表
    for k in range(num_EMO):
        AU_pos_nonzero = np.nonzero(EMO2AU_np[k])[0]  #EMO=k的AU-EMO关联（非0值）的位置
        AU_pos_certain = np.where(EMO2AU_np[k]==1)[0]  #EMO=k的AU-EMO关联（1值）的位置
        for j in range(len(AU_pos_nonzero)):
            if EMO2AU_np[k][AU_pos_nonzero[j]] == 1.0:  #当AUj是确定发生的时候，AUi同时发生的概率就是AUi自己本身的值
                for i in range(len(AU_pos_nonzero)):
                    if i != j:
                        AU_cpt_tmp[AU_pos_nonzero[i]][AU_pos_nonzero[j]] += EMO2AU_np[k][AU_pos_nonzero[i]]
            else:  #而当AUj的发生是不确定的时候，只初始化确定发生的AUi，值为1
                for i in range(len(AU_pos_certain)):
                    AU_cpt_tmp[AU_pos_certain[i]][AU_pos_nonzero[j]] += 1.0
    
    AU_cpt = (AU_cpt_tmp / num_EMO) + np.eye(num_AU)
    EMO2AU_cpt = EMO2AU_np
    AU = list(map(int, AU))
    return EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU


if __name__ == '__main__':
    init_EMO2AU_cpt, init_prob_AU, init_EMO_img_num, init_AU_cpt, EMO, AU = cal_interAUPriori()