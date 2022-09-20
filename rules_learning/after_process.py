from email.mime import base
import os
import numpy as np
import pickle as pkl
from materials.process_priori import cal_interAUPriori
import csv
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


def draw_EMO2AU(base_path, EMO2AU, new_AU_cpt):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    str_AU = ['AU'+str(i) for i in AU]

    EMO2AU_df = pd.DataFrame(EMO2AU, index=EMO, columns=str_AU)
    f, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(EMO2AU_df, annot=True, cmap="YlGnBu", fmt='.3f', ax = ax)
    fig_path = os.path.join(base_path, 'new_EMO2AU.jpg')
    plt.savefig(fig_path, dpi=500)
    plt.close()

    interAU_df = pd.DataFrame(new_AU_cpt, index=str_AU, columns=str_AU)
    f, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(interAU_df, annot=True, cmap="YlGnBu", fmt='.3f', ax = ax)
    fig_path = os.path.join(base_path, 'new_AU_cpt.jpg')
    plt.savefig(fig_path, dpi=500)
    plt.close()

    path = os.path.join(base_path, 'trained.xlsx')
    file_open= pd.ExcelWriter(path)
    EMO2AU_df.to_excel(file_open, sheet_name='EMO2AU')
    interAU_df.to_excel(file_open, sheet_name='ineterAU')
    file_open.save()

    for i in range(EMO2AU.shape[0]):
        cur_EMO2AU = np.argsort(-EMO2AU[i, :])[:5]
        print('AU occ of %s (from high to low): '%(EMO[i]), end='')
        for j in range(cur_EMO2AU.shape[0]):
            if j != cur_EMO2AU.shape[0]-1:
                print('AU'+str(AU[cur_EMO2AU[j]]), end=', ')
            else:
                print('AU'+str(AU[cur_EMO2AU[j]]))
    a = 1
    

def draw_interAUModel():
    base_path = '/media/data1/wf/AU_EMOwPGM/codes/save'
    model_path = 'newEMO2AUwP(AU)_staticAU256_occAU!=0_LR=0.001+0.0001at20000_AllMultiFactor_interAUwPGM'
    PGM_path = os.path.join(base_path, model_path)
    pkl_path = os.path.join(PGM_path, 'results.pkl')
    with open(pkl_path, 'rb') as fo:
        pkl_file = pkl.load(fo)
    interAU_model = pkl_file['interAU_model']

    G = nx.DiGraph()
    nodes = interAU_model.nodes
    G.add_nodes_from(nodes)
    edges = interAU_model.edges
    G.add_edges_from(edges)

    # pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)
    # nx.draw(G, pos)

    node_labels = {}
    for node in G.nodes:
        node_labels[node] = node

    nx.draw_networkx(G, pos, vmin=10, vmax=20, width=2, linewidths=0.7, font_size=7, edge_color='black')
    plt.show()
    fig_path = os.path.join(base_path, 'input_interAU.jpg')
    plt.savefig(fig_path, dpi=500)
    a = 1


if __name__ == '__main__':
    base_path = '/media/data1/wf/AU_EMOwPGM/codes/save/BP4D/2022-07-30'
    pkl_path = os.path.join(base_path, 'results.pkl')
    with open(pkl_path, 'rb') as fo:
        pkl_file = pkl.load(fo)
    EMO2AU = pkl_file['new_EMO2AU']
    new_AU_cpt = pkl_file['new_AU_cpt']
    draw_EMO2AU(base_path, EMO2AU, new_AU_cpt)

    # draw_interAUModel()

    a = 1
