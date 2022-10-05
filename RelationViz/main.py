import sys
# sys.path.append('/media/data1/wf/AU_EMOwPGM/codes')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
import seaborn as sns

from py2neo import Graph, Node, Relationship, NodeMatcher
from py2neo.matching import RelationshipMatcher

def draw_EMO2AU(graph, rules):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = rules

    interAU_thresh = 0.05
    EMO2AU_thresh = 0.5
    # emo_i = 0
    # emo = EMO[emo_i]
    for emo_i, emo in enumerate(EMO):
        last_prob = 0

        emo_node = Node("emotion", name=emo)
        graph.create(emo_node)
        
        cur_EMO2AU = np.argsort(-EMO2AU_cpt[emo_i, :])
        num_AU = len(cur_EMO2AU)
        last_layer = [(-1, emo_node)]
        new_layer = []
        for occ_i in range(num_AU):
            au_i = cur_EMO2AU[occ_i]
            cur_prob = EMO2AU_cpt[emo_i][au_i]
            if cur_prob >= EMO2AU_thresh:
                au_node_name = 'AU'+str(AU[au_i])
                au_node = Node("AU", name=au_node_name)
                graph.create(au_node)
                
                if occ_i == 0:
                    r_value = str(cur_prob)[:5]
                    new_layer.append((au_i, au_node))
                    # graph.create(Relationship(emo_node, r_value, au_node))
                    r = Relationship(emo_node, 'EMO2AU', au_node)
                    r['value'] = r_value
                    graph.create(r)
                    last_prob = cur_prob
                else:
                    if last_prob - cur_prob <= interAU_thresh:
                        new_layer.append((au_i, au_node))
                    else:
                        last_layer = new_layer
                        new_layer = []
                        new_layer.append((au_i, au_node))
                        last_prob = cur_prob

                    for last_node_i, last_node in last_layer:
                        if last_node_i == -1:
                            r_value = str(EMO2AU_cpt[emo_i][au_i])[:5]
                            # graph.create(Relationship(emo_node, r_value, au_node))
                            r = Relationship(emo_node, 'EMO2AU', au_node)
                            r['value'] = r_value
                            graph.create(r)
                        else:
                            r_value = str(AU_cpt[au_i][last_node_i])[:5]
                            # graph.create(Relationship(last_node, r_value, au_node))
                            r = Relationship(last_node, 'interAU', au_node)
                            r['value'] = r_value
                            graph.create(r)

                            r_value = str(EMO2AU_cpt[emo_i][au_i])[:5]
                            r = Relationship(emo_node, 'EMO2AU', au_node)
                            r['value'] = r_value
                            graph.create(r)


    # for relation in all_relation:
    #     graph.create(relation)

    end_flag = 1

def draw_AU2EMO(graph, rules):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = rules
    AU2EMO_cpt = np.zeros(EMO2AU_cpt.T.shape)
    for i, au in enumerate(AU):
        for j, emo in enumerate(EMO):
            AU2EMO_cpt[i, j] = EMO2AU_cpt[j, i] / len(EMO) / prob_AU[i] # 注意此处的 /len(EMO) 表达的其实是p(EMO)，此处存在各个类别均匀分布的假设

    AU2EMO_df = pd.DataFrame(AU2EMO_cpt.T, index=EMO, columns=['AU'+str(i) for i in AU])
    # plot_heatmap(AU2EMO_df)

    interAU_thresh = 0.05
    AU2EMO_thresh = 0.2
    thresh = 0.05
    # emo_i = 0
    # emo = EMO[emo_i]
    for emo_i, emo in enumerate(EMO):
        last_prob = 0

        emo_node = Node("emotion", name=emo)
        graph.create(emo_node)
        
        cur_AU2EMO = np.argsort(-AU2EMO_cpt[:, emo_i])
        print('AU occ of %s (from high to low): '%(EMO[emo_i]), end='')
        for j in range(cur_AU2EMO.shape[0]):
            if j != cur_AU2EMO.shape[0]-1:
                print('AU'+str(AU[cur_AU2EMO[j]]), end=', ')
            else:
                print('AU'+str(AU[cur_AU2EMO[j]]))
        
        num_AU = len(cur_AU2EMO)

        first_node = 'AU' + str(AU[cur_AU2EMO[0]])
        au_node = Node("AU", name=first_node)
        graph.create(au_node)
        r_value = str(AU2EMO_cpt[cur_AU2EMO[0]][emo_i])[:5]
        r = Relationship(au_node, 'AU2EMO', emo_node)
        r['value'] = r_value
        graph.create(r)

        new_layer = [(cur_AU2EMO[0], au_node)]
        last_prob = AU2EMO_cpt[cur_AU2EMO[0]][emo_i]
        last_layer = []

        for au_i_tmp in range(1, num_AU):
            au_i = cur_AU2EMO[au_i_tmp]
            cur_prob = AU2EMO_cpt[au_i][emo_i]
            if cur_prob >= AU2EMO_thresh:
                au_node_name = 'AU'+str(AU[au_i])
                au_node = Node("AU", name=au_node_name)
                graph.create(au_node)
                
                if last_prob - cur_prob <= interAU_thresh:
                    new_layer.append((au_i, au_node))
                else:
                    last_layer = new_layer
                    new_layer = []
                    new_layer.append((au_i, au_node))
                    last_prob = cur_prob

                if len(last_layer) != 0:
                    for last_node_i, last_node in last_layer:
                        r_value = str(AU_cpt[au_i][last_node_i])[:5]
                        r = Relationship(last_node, 'interAU', au_node)
                        r['value'] = r_value
                        graph.create(r)

                r_value = str(AU2EMO_cpt[au_i][emo_i])[:5]
                r = Relationship(au_node, 'AU2EMO', emo_node)
                r['value'] = r_value
                graph.create(r)

        # last_layer = new_layer
        # for last_node_i, last_node in last_layer:

        #     r_value = str(AU2EMO_cpt[au_i][emo_i])[:5]
        #     r = Relationship(au_node, 'AU2EMO', emo_node)
        #     r['value'] = r_value
        #     graph.create(r)

        end_flag = True

    end_flag = True

def draw_all_AU2EMO(graph, rules):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = rules
    AU2EMO_cpt = np.zeros(EMO2AU_cpt.T.shape)
    for i, au in enumerate(AU):
        for j, emo in enumerate(EMO):
            AU2EMO_cpt[i, j] = EMO2AU_cpt[j, i] / len(EMO) / prob_AU[i] # 注意此处的 /len(EMO) 表达的其实是p(EMO)，此处存在各个类别均匀分布的假设

    AU_nodes = []
    for au_i, au in enumerate(AU):
        node = 'AU' + str(au)
        au_node = Node("AU", name=node)
        graph.create(au_node)
        AU_nodes.append(au_node)

    interAU_thresh = 0.05
    AU2EMO_thresh = 0.2
    thresh = 0.05
    # emo_i = 0
    # emo = EMO[emo_i]
    for emo_i, emo in enumerate(EMO):
        last_prob = 0

        emo_node = Node("emotion", name=emo)
        graph.create(emo_node)
        
        cur_AU2EMO = np.argsort(-AU2EMO_cpt[:, emo_i])
        print('AU occ of %s (from high to low): '%(EMO[emo_i]), end='')
        for j in range(cur_AU2EMO.shape[0]):
            if j != cur_AU2EMO.shape[0]-1:
                print('AU'+str(AU[cur_AU2EMO[j]]), end=', ')
            else:
                print('AU'+str(AU[cur_AU2EMO[j]]))
        num_AU = len(cur_AU2EMO)
        
        au_node = AU_nodes[cur_AU2EMO[0]]
        r_value = str(AU2EMO_cpt[cur_AU2EMO[0]][emo_i])[:5]
        r = Relationship(au_node, 'AU2EMO', emo_node)
        r['value'] = r_value
        graph.create(r)

        new_layer = [(cur_AU2EMO[0], au_node)]
        last_prob = AU2EMO_cpt[cur_AU2EMO[0]][emo_i]
        last_layer = []

        for au_i_tmp in range(1, num_AU):
            au_i = cur_AU2EMO[au_i_tmp]
            cur_prob = AU2EMO_cpt[au_i][emo_i]
            if cur_prob >= AU2EMO_thresh:
                au_node = AU_nodes[au_i]
                graph.create(au_node)
                
                if last_prob - cur_prob <= interAU_thresh:
                    new_layer.append((au_i, au_node))
                else:
                    last_layer = new_layer
                    new_layer = []
                    new_layer.append((au_i, au_node))
                    last_prob = cur_prob

                # if len(last_layer) != 0:
                #     for last_node_i, last_node in last_layer:
                #         r_value = str(AU_cpt[au_i][last_node_i])[:5]
                #         r = Relationship(last_node, 'interAU', au_node)
                #         r['value'] = r_value
                #         graph.create(r)

                r_value = str(AU2EMO_cpt[au_i][emo_i])[:5]
                r = Relationship(au_node, 'AU2EMO', emo_node)
                r['value'] = r_value
                graph.create(r)

        # last_layer = new_layer
        # for last_node_i, last_node in last_layer:

        #     r_value = str(AU2EMO_cpt[au_i][emo_i])[:5]
        #     r = Relationship(au_node, 'AU2EMO', emo_node)
        #     r['value'] = r_value
        #     graph.create(r)

        end_flag = True

    end_flag = True

def draw_continuous(graph):
    info_path = 'F:/wf/AU_EMOwPGM/codes/results0911_theTrust/BP4D/Test/subject_independent/bs_64_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/continuous/all_done.pth'
    info = torch.load(info_path, map_location='cpu')
    dataset_order = info['conf.dataset_order']
    for dataset_i, dataset_name in enumerate(dataset_order[1:]):
        print('/n')
        graph.delete_all()
        key = 'rules_' + dataset_name
        rules = info[key]
        draw_all_AU2EMO(graph, rules)
        end_flag = 1
    end_flag = 1

def plot_heatmap(df, save_path=None, img_name='heat.jpg'):
    f, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt='.3f', ax = ax)
    if save_path is None:
        save_path = 'F:/wf/AU_EMOwPGM/codes/results'
    fig_path = os.path.join(save_path, img_name)
    plt.savefig(fig_path, dpi=500)
    plt.close()

def call_AU2EMO(rules):
    EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = rules
    AU2EMO_cpt = np.zeros(EMO2AU_cpt.T.shape)
    for i, au in enumerate(AU):
        for j, emo in enumerate(EMO):
            AU2EMO_cpt[i, j] = EMO2AU_cpt[j, i] / len(EMO) / prob_AU[i] # 注意此处的 /len(EMO) 表达的其实是p(EMO)，此处存在各个类别均匀分布的假设
    return AU2EMO_cpt

def tmp(all_info):
    priori_rules = all_info['priori_rules']
    priori_AU2EMO = call_AU2EMO(priori_rules)
    latest_rules = all_info['latest_rules']
    latest_AU2EMO = call_AU2EMO(latest_rules)
    to_RAF_rules = all_info['rules_RAF-DB']
    to_RAF_AU2EMO = call_AU2EMO(to_RAF_rules)
    to_AffectNet_rules = all_info['rules_AffectNet']
    to_AffectNet_AU2EMO = call_AU2EMO(to_AffectNet_rules)
    a = 1

def main():
    graph = Graph("http://localhost:7474", username="neo4j", password='123456') # 连接数据库
    graph.delete_all()

    # pre_path = 'F:/wf/AU_EMOwPGM/codes/results/AffectNet/Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001'
    # info_type_path = 'predsAU_labelsEMO'
    # info_source_path = 'epoch1_model_fold0'
    # rules_path = os.path.join(pre_path, info_type_path, info_source_path + '.pth')
    # rules = torch.load(rules_path, map_location='cpu')['output_rules']
    # EMO2AU_cpt, AU_cpt, prob_AU, ori_size, num_all_img, AU_ij_cnt, AU_cnt, EMO, AU = rules

    info_path = 'F:/wf/AU_EMOwPGM/codes/results/BP4D/Test/subject_independent/bs_128_seed_0_lrEMO_0.0003_lrAU_0.0001_lr_relation_0.001/continuous_v2/all_done.pth'
    all_info = torch.load(info_path, map_location='cpu')
    rules = all_info['rules_RAF-DB']

    # draw_EMO2AU(graph, rules)

    # draw_AU2EMO(graph, rules)

    # draw_continuous(graph)

    tmp(all_info)

    end_flag = 1

    

if __name__ == '__main__':
    main()


'''
# EMO 节点
    EMO_nodes = []
    for emo_i, emo in enumerate(EMO):
        emo_node = Node("emotion", name=emo)
        EMO_nodes.append(emo_node)
    
    # AU 节点
    AU_nodes = []
    for au_i, au in enumerate(AU):
        au_node = Node("AU", name=au)
        AU_nodes.append(au_node)
'''
