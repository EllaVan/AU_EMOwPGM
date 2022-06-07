import argparse
import numpy as np
import pickle as pkl
from materials.process_priori import cal_interAUPriori
from models.AU_EMO_BP import UpdateGraph
import torch
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------basic settings------------------------
    parser.add_argument('--gpu', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='BP4D')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = utils.getDatasetInfo(args.dataset)
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = tuple(train_loader.dataset.priori.values())

    AU_EMO_jpt = EMO2AU_cpt*1.0/len(EMO)
    AU2EMO_cpt = []
    for i in range(len(AU)):
        AU2EMO_cpt.append(list(AU_EMO_jpt[:, i]/prob_AU[i]))
    AU2EMO_cpt = np.array(AU2EMO_cpt)
    AU2EMO_cpt = np.where(AU2EMO_cpt >= 0, AU2EMO_cpt, 1e-4)

    pkl_path = '/media/database/data4/wf/AU_EMOwPGM/codes/save/2022-06-07/results.pkl'
    with open(pkl_path, 'rb') as fo:
        pkl_file = pkl.load(fo)
    ori_AU2EMO = AU2EMO_cpt
    new_AU2EMO = pkl_file['new_AU2EMO']
    oriACC = []
    newACC = []
    for k, emo in enumerate(EMO):
        test_data = test_loader[k]
        ori_acc = []
        new_acc = []
        for idx, (cur_item, emo_label, index) in enumerate(train_loader, 1):
            x = []
            ori_weight = []
            new_weight = []
            au_item = []
            
            for i, au in enumerate(AU):
                if cur_item[0, au] == 1:
                    x.append(1.0)
                    au_item.append(i)
                    ori_weight.append(ori_AU2EMO[i, :])
                    new_weight.append(new_AU2EMO[i, :])
            if emo_label == 0:
                x.append(1.0)
                ori_weight.append(ori_AU2EMO[-2, :])
                new_weight.append(new_AU2EMO[-2, :])
            elif emo_label == 2 or emo_label == 4:
                x.append(1.0)
                x.append(1.0)
                ori_weight.append(ori_AU2EMO[-1, :])
                new_weight.append(new_AU2EMO[-1, :])
                ori_weight.append(ori_AU2EMO[-2, :])
                new_weight.append(new_AU2EMO[-2, :])
            elif emo_label == 5:
                x.append(1.0)
                ori_weight.append(ori_AU2EMO[-1, :])
                new_weight.append(new_AU2EMO[-1, :])

            if len(x) != 0:
                ori_weight = np.array(ori_weight)
                new_weight = np.array(new_weight)
                ori_update = UpdateGraph(in_channels=1, out_channels=len(EMO), W=ori_weight).to(device)
                new_update = UpdateGraph(in_channels=1, out_channels=len(EMO), W=new_weight).to(device)

                AU_evidence = torch.ones((1, 1)).to(device)
                ori_prob = ori_update(AU_evidence)
                ori_pred = torch.argmax(ori_prob)
                new_prob = new_update(AU_evidence)
                new_pred = torch.argmax(new_prob)

                ori_acc.append(torch.eq(ori_pred, emo_label.to(device)).type(torch.FloatTensor).item())
                oriACC.append(torch.eq(ori_pred, emo_label.to(device)).type(torch.FloatTensor).item())
                new_acc.append(torch.eq(new_pred, emo_label.to(device)).type(torch.FloatTensor).item())
                newACC.append(torch.eq(new_pred, emo_label.to(device)).type(torch.FloatTensor).item())

        print('The ori Acc of %s is %.5f, and the new Acc is %.5f' 
                %(emo, np.array(ori_acc).mean(), np.array(new_acc).mean()))
    print('The ori Acc of the dataset is %.5f, and the new Acc is %.5f' 
            %(np.array(oriACC).mean(), np.array(newACC).mean()))

    pass