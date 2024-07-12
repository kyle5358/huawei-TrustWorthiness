import networkx as nx
from tqdm import tqdm
import torch 
import pdb 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def binarySearch(arr, l, r, x):
    # 基本判断
    if r >= l:
        mid = int(l + (r - l) / 2)
        # 元素整好的中间位置
        if arr[mid] >= x and arr[mid-1]<x:
            # print([arr[mid-1],arr[mid]])
            return mid
            # 元素小于中间位置的元素，只需要再比较左边的元素
        elif x< arr[mid-1] :
            return binarySearch(arr, l, mid - 1, x)
            # 元素大于中间位置的元素，只需要再比较右边的元素
        else:             #x>arr[mid]
            return binarySearch(arr, mid+1 , r, x)

def get_data_txt(trainfile):
    train_triple = []
    train_confidence = []

    f = open(trainfile, "r")
    lines = f.readlines()
    for line in lines:
        tri = line.rstrip('\r\n').rstrip('\n').rstrip('\r').split(' ')
        train_triple.append((int(tri[0]), int(tri[1]), int(tri[2]), int(tri[3])))
        if tri[3] == '1':
            train_confidence.append([0, 1])
        else:
            train_confidence.append([1, 0])
    f.close()

    return train_triple, train_confidence

import numpy
def text2dg(subgraph1dir):  # 将具体节点替换为节点的类别
    # m=numpy.array([0,341, 351, 409, 10980, 11364, 11659, 11853, 12055, 12141, 13254, 13268, 13309, 13311, 13313, 13327, 13358, 14246, 14416, 14600, 14616, 18458, 19438, 20819, 20853, 20910, 20925, 21066, 21080, 21097, 21104, 21200, 21206, 34896]  )
    m=  numpy.array([0,341, 351, 409, 10980, 11328, 11615, 11807, 11981, 12058, 13166, 13180, 13221, 13223, 13225, 13239, 13270, 14140, 14309, 14493, 14508, 14568, 16929, 16963, 17020, 17035, 17176, 17190, 17207, 17214, 17310, 17316, 25927]  )
    with open(subgraph1dir, 'r') as f1:
        leigh_catego = {}
        first_line = f1.readline().strip()
        nodes = first_line.split('\t')
        g1 = nx.Graph()
        for node in nodes:
            category = binarySearch(m, 1, 33, int(node))
            nodename = 'categ' + str(category)
            if nodename not in leigh_catego.keys():
                leigh_catego[nodename] = [node]
            else:
                leigh_catego[nodename].append(node)

                # next(f1)
        # for line_number, line in enumerate(f1, start=1):
        for line in f1:
            list = line.split("\t")
            for key, value in leigh_catego.items():
                if list[1] in value:
                    g1.add_edge(list[0], key)
    return g1

def get_editDistance(train_triple):
    edit_dict=[]
    filedir='/data1/yk/examine/huawei/TrustWorthiness/dataset/2part_subGraphs_1/'
    correspond = {}#{1: {0: [13552, 14121], 1: [21198, 21198]}, 2: {0: [12   (0:头实体，1：尾实体)
    for triple in train_triple:
        if triple[2] not in correspond.keys():
            correspond[triple[2]] = {}
            correspond[triple[2]][0] = [triple[0]]
            correspond[triple[2]][1] = [triple[1]]
        else:
            correspond[triple[2]][0].append(triple[0])
            correspond[triple[2]][1].append(triple[1])
    #print(correspond)

    for triple in tqdm(train_triple):
        #print(triple)
        h_edit_sum=0.0
        t_edit_sum=0.0
        h1dir=filedir+str(triple[0])+'.txt'
        h1_subgraph=text2dg(h1dir)
        t1dir=filedir+str(triple[1])+'.txt'
        t1_subgraph = text2dg(t1dir)

        Random=numpy.random.randint(0,len(correspond[triple[2]][0]))
        h2=correspond[triple[2]][0][Random]
        h2dir=filedir+str(h2)+'.txt'
        h2_subgraph=text2dg(h2dir)
        tmp1 = h1_subgraph
        tmp2 = h2_subgraph
        if tmp1.number_of_edges() > tmp2.number_of_edges():
            tmp = tmp1
            tmp1 = tmp2
            tmp2 = tmp
        tmph = nx.graph_edit_distance(tmp1, tmp2)
        h_edit_sum=tmph

        t2=correspond[triple[2]][1][Random]
        t2dir=filedir+str(t2)+'.txt'
        t2_subgraph = text2dg(t2dir)
        tmp1 = t1_subgraph
        tmp2 = t2_subgraph
        if tmp1.number_of_edges() > tmp2.number_of_edges():
            tmp = tmp1
            tmp1 = tmp2
            tmp2 = tmp
        tmpt = nx.graph_edit_distance(tmp1, tmp2)
        t_edit_sum = tmpt

        GED_h=h_edit_sum
        GED_t=t_edit_sum
        aver=(GED_h+GED_t)/2
        aver=1.0 / (1 + numpy.exp(-aver))
        edit_dict.append(aver)
        
    edit_dict=torch.tensor(edit_dict,dtype=torch.float32)
    mean=torch.mean(edit_dict)
    std=torch.std(edit_dict)
    tmp=(edit_dict-mean)/std
    
    return tmp
