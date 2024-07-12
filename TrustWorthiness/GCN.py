import numpy
import networkx as nx
import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from torch_geometric.data import Data as gData

#得到头尾实体的一阶邻居子图
# def text2dg(subgraph1dir):  # 将具体节点替换为节点的类别
#     with open(subgraph1dir, 'r') as f1:
#         first_line = f1.readline().strip()
#         nodes = first_line.split('\t')
#         g1 = nx.Graph()

#         for node in nodes[1:]:
#             g1.add_edge(nodes[0], node)   
#     return g1

#节点分类任务
#定义模型
# def text2dg(subgraph1dir):  # 将具体节点替换为节点的类别
#     #edge_index=[]
#     node_index=0
#     array_in=np.array([])
#     array_out=np.array([])
#     with open(subgraph1dir, 'r') as f1:
#         first_line = f1.readline().strip()
#         nodes = first_line.split('\t')
#         print(nodes)
#         nums=len(nodes)

#         for i in range(nums):
#             np.append(array_in,0)
#             np.append(array_out,i+1)
#             np.append(array_in,i+1)
#             np.append(array_out,0)
#     edge_index=torch.cat((torch.tensor(array_in,dtype=torch.int).unsqueeze(1),torch.tensor(array_out,dtype=torch.int).unsqueeze(1)), dim=-1)
#     print(array_in)
#     print(array_out)
#     return edge_index
import pickle
# datafile = "../model/data2_TransE.pkl"  # 装数据
# datafile = "/yq/yk/huawei/model/data2_TransE.pkl"  # 装数据
# ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
#     entity2vec, entity2vec_dim, \
#     relation2vec, relation2vec_dim, \
#     train_triple, train_confidence,train_triple_vec, \
#     test_triple, test_confidence, \
#     train_transE, test_transE,\
#     train_ComplEx,test_ComplEx,\
#     _,_,_,_= pickle.load(open(datafile, 'rb'))

def text2dg(subgraph1dir,entity2vec):
    array_in = np.array([])
    array_out = np.array([])
    tmp=[]
    tmpx=[]
    with open(subgraph1dir, 'r') as f1:
        first_line = f1.readline().strip()
        nodes = first_line.split('\t')
        #print(nodes)
        nums = len(nodes)

        # for i in range(1,nums):
        #     array_in = np.append(array_in, 0)
        #     array_out = np.append(array_out, i )
        #     array_in = np.append(array_in, i )
        #     array_out = np.append(array_out, 0)
        if nums >=20:#最多有20个节点，19个邻居
            nums=20
            x=np.zeros(nums-1)
            y=np.arange(1,nums)
            array_in=np.concatenate((x, y), axis=0)
            array_out=np.concatenate((y, x), axis=0)

            for node in nodes[0:20]:#0到19
                tmpx.append(torch.tensor(entity2vec[int(node)],dtype=torch.float32))
            x=torch.stack(tmpx)
        else:
            x=np.zeros(nums-1)
            y=np.arange(1,nums)
            z=19*np.ones(40-2*nums)
            array_in=np.concatenate((x, y,z), axis=0)
            array_out=np.concatenate((y, x,z), axis=0)            
            for node in nodes:#0到19 #nums*100
                tmpx.append(torch.tensor(entity2vec[int(node)],dtype=torch.float32))

            com=20-nums
            com_matrix=torch.zeros(com,100)
            tmpx=torch.stack(tmpx)
            x=torch.concatenate((tmpx, com_matrix), axis=0)
            
        # print(tmpx)
        # print(torch.is_tensor(tmpx))
    tmp.append(torch.tensor(array_in,dtype=torch.int))
    tmp.append(torch.tensor(array_out,dtype=torch.int))
    edge_index=torch.stack(tmp)#[[],[]]2*(node_num-1)
    #x=torch.stack(tmpx)#[[],[],[]] [num_node,node_dim]
    #print(edge_index)
    
    return edge_index,x

from tqdm import tqdm
def get_gcn(train_triple,graphfolder_dir,entity2vec):
    # train1_edge_index=[]
    # train2_edge_index=[]
    # train1_nodex=[]
    # train2_nodex=[]
    train1_datalist=[]
    train2_datalist=[]
    for triple in tqdm(train_triple):
        h_path=graphfolder_dir+str(triple[0])+'.txt'
        edge_index,x=text2dg(h_path,entity2vec)
        # train1_edge_index.append(torch.tensor(edge_index,dtype=torch.int))
        # train1_nodex.append(torch.tensor(x,dtype=torch.float32))
        train1_datalist.append(gData(x=x,edge_index=edge_index))

        t_path=graphfolder_dir+str(triple[1])+'.txt'
        edge_index,x=text2dg(t_path,entity2vec)
        train2_datalist.append(gData(x=x,edge_index=edge_index))
        # train2_edge_index.append(torch.tensor(edge_index,dtype=torch.int))
        # train2_nodex.append(torch.tensor(x,dtype=torch.float32))

    # return train1_edge_index,train1_nodex,train2_edge_index,train2_nodex
    return train1_datalist,train2_datalist
# 示例用法

#x 节点特征矩阵，edge_index,y
class GCN(nn.Module):
    def __init__(self, num_node_features,num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, edge_index,x):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)[0]
    
# # # 搭建神经网络
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.gcnh=GCN(100,2)
#         self.gcnt=GCN(100,2)
#         self.model = nn.Sequential(
#             nn.Linear(6, 100),
#             nn.Dropout(0.5),
#             nn.Linear(100, 2),#输出维度修改
#             nn.Sigmoid()
#         )

#     def forward(self,train1_edge_index,train1_nodex,train2_edge_index,train2_nodex,transE_input,ComplEx_input):
#         gcnh_out=self.gcnh(train1_edge_index,train1_nodex)
#         gcnt_out=self.gcnt(train2_edge_index,train2_nodex)
#         BP_input = torch.cat((transE_input.unsqueeze(1),ComplEx_input.unsqueeze(1),gcnh_out,gcnt_out), dim=-1)
#         x = self.model(BP_input)
#         return x
    
# dir="/yq/yk/huawei/dataset/subGraphs_1/25872.txt"
# G=text2dg(dir)
# print("Nodes in G:", G.nodes())
# print("Edges in G:", G.edges())
if __name__ == "__main__":
    model=GCN(100,2)
    dir="/yq/yk/huawei/dataset/subGraphs_1/25872.txt"
    edge_index,x=text2dg(dir,entity2vec)
    print(model(edge_index,x))
    # print(edge_index.shape)
    # print(x.shape)
    # print("edge_index:{}".format(torch.is_tensor(edge_index)))
    # print("x:{}".format(torch.is_tensor(x)))

    dir="/yq/yk/huawei/dataset/subGraphs_1/25027.txt"
    edge_index,x=text2dg(dir,entity2vec)
    print(model(edge_index,x))
    # print(edge_index.shape)
    # print(x.shape)
    # print("edge_index:{}".format(torch.is_tensor(edge_index)))
    # print("x:{}".format(torch.is_tensor(x)))
    # my=GCN(100,2)
    # out=my(edge_index,x)
    # print(out)