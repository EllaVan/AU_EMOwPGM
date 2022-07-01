import os
import numpy as np
import pickle as pkl
from materials.process_priori import cal_interAUPriori
import csv
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


def draw_EMO2AU(base_path, EMO2AU):
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    df = pd.DataFrame(EMO2AU, index=EMO, columns=AU)

    # csv_path = os.path.join(base_path, 'new_EMO2AU.csv')
    # df.to_csv(csv_path, encoding='utf-8-sig')
    # with open(csv_path, 'w', encoding='UTF8', newline='') as f: # 写入PGM推理的结果
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerows(a)
    
    f, ax = plt.subplots(figsize=(26,5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt='.3f', ax = ax)
    fig_path = os.path.join(base_path, 'new_EMO2AU.jpg')
    plt.savefig(fig_path, dpi=500)
    plt.show()

    for i in range(EMO2AU.shape[0]):
        cur_EMO2AU = np.argsort(-EMO2AU[i, :])[:5]
        print('AU occ of %s (from high to low): '%(EMO[i]), end='')
        for j in range(cur_EMO2AU.shape[0]):
            if j != cur_EMO2AU.shape[0]-1:
                print('AU'+str(AU[cur_EMO2AU[j]]), end=', ')
            else:
                print('AU'+str(AU[cur_EMO2AU[j]]))
    a = 1
    

def draw_interAU():
    base_path = '/media/database/data4/wf/AU_EMOwPGM/codes/save/2022-06-30'
    pkl_path = os.path.join(base_path, 'results.pkl')
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

    nx.draw_networkx(G, pos, vmin=10, vmax=20, width=2, font_size=8, edge_color='black')
    plt.show()
    fig_path = os.path.join(base_path, 'input_interAU.jpg')
    plt.savefig(fig_path, dpi=500)
    a = 1


if __name__ == '__main__':
    base_path = '/media/database/data4/wf/AU_EMOwPGM/codes/save/2022-06-30'
    pkl_path = os.path.join(base_path, 'results.pkl')
    with open(pkl_path, 'rb') as fo:
        pkl_file = pkl.load(fo)

    EMO2AU = pkl_file['new_EMO2AU']
    draw_EMO2AU(base_path, EMO2AU)

    # draw_interAU()
