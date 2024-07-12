#显示路径置信度
import os
import numpy
# from utils import get_data_txt
from tqdm import tqdm
import numpy as np
import math
import torch

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

#实体变为概念实体，关系序号不变
def extract(string):
    tmp = string.strip('()').replace(' ','').split(',')
    m= numpy.array([0,341, 351, 409, 10980, 11328, 11615, 11807, 11981, 12058, 13166, 13180, 13221, 13223, 13225, 13239, 13270, 14140, 14309, 14493, 14508, 14568, 16929, 16963, 17020, 17035, 17176, 17190, 17207, 17214, 17310, 17316, 25927]  )
    inverted_list = [str(binarySearch(m,1,32,int(item))) for item in tmp[:2]]+tmp[2:]
    return inverted_list

# file_path="./95_121_6.txt"

#entity_chain:{'1_1_1_1': {'6': 10}, '1_1_1': {'6': 4}, '1_1': {'6': 1}}
#统计entity_chain
def get_entity_chain(folder_path):
    entity_chain={}
    for filename in os.listdir(folder_path):
        #对每个路径文件
        guard=[]
        with open(os.path.join(folder_path, filename), "r") as f:
            #对路径txt文件的每一行
            for line in f:
                #print(line.strip())
                tmp=line.strip().split('\t')
                string=extract(tmp[0])[0]+"_"+extract(tmp[0])[1]
                for i in range(1,len(tmp)-1):
                    string+="_"+extract(tmp[i])[1]
                
                #filename=os.path.basename(file_path)
                parts = filename.split('_')
                relation = parts[-1].split('.')[0]
                # print("relation：{}".format(relation))
                
                if string in guard:
                    continue
                if string in entity_chain.keys():
                    if relation in entity_chain[string].keys():
                        entity_chain[string][relation]+=1
                    else:
                        entity_chain[string][relation]=1
                else:
                    entity_chain[string]={}
                    entity_chain[string][relation]=1     
                guard.append(string)
    return entity_chain
# print("entity_chain:{}".format(entity_chain))

#统计r出现的总数
def cal_relation(trainfile,testfile):
    relation_dict={}
    f = open(trainfile, "r")
    lines = f.readlines()
    for line in lines:
        tri = line.rstrip('\r\n').rstrip('\n').rstrip('\r').split(' ')
        if tri[2] in relation_dict.keys():
            relation_dict[tri[2]]+=1
        else:
            relation_dict[tri[2]]=1
    f.close()
    return relation_dict

# relation_dict:{'1': 196, '2': 46292, '3': 55120, '4': 178, '5': 19496, '6': 1782, '7': 1174, '8': 144, '9': 922, '10': 120, '11': 26, '12': 2210, '13': 1248, '14': 14, '15': 50, '16': 2208, '17': 546, '18': 3834, '19': 150, '20': 544, '21': 24, '22': 96, '23': 276, '24': 72, '25': 94, '26': 306, '27': 296, '28': 27022, '29': 700, '30': 502, '31': 1160, '32': 400, '33': 312}

#entity_chain:{'1_1_1_1': {'6': 10}, '1_1_1': {'6': 4}, '1_1': {'6': 1}}
def tf_idf(entity_chain,train_triple,relation_dict,folder_path):
    PathScore=[]
    print("loading tfidf")
    for tri in tqdm(train_triple):
        tri0=str(tri[0])
        tri1=str(tri[1])
        tri2=str(tri[2])
        #打开三元组的路径文件
        text_path=folder_path + str(tri[0]) + '_' + str(tri[1]) + '_' + str(tri[2]) + '.txt'
        sum=0#·每个三元组，路径文件的权重之和
        if os.path.exists(text_path) is True:
            if os.path.getsize(text_path) != 0:
                with open(os.path.join(text_path), "r") as f:
                    #对路径txt文件的每一行,获取概念路径，计算一个得分
                    for line in f:
                        tmp=line.strip().split('\t')
                        string=extract(tmp[0])[0]+"_"+extract(tmp[0])[1]
                        for i in range(1,len(tmp)-1):
                            string+="_"+extract(tmp[i])[1]           
                
                        cooc=entity_chain[string][tri2]
                        rel_oc=relation_dict[tri2]
                        total_rel_cat=len(relation_dict)
                        co_rel_cat=len(entity_chain[string])+1
                        score=float(cooc*total_rel_cat)/(float(rel_oc*co_rel_cat))
                        sum+=score
            else:#只有单条路径，且因为某种原因为写入文件
                #创造string
                
                rel_oc = relation_dict[tri2]

                rel_oc=relation_dict[tri2]
                total_rel_cat=len(relation_dict)
                sum=0.5*float(total_rel_cat)/float(rel_oc)  
        else:
            rel_oc=relation_dict[tri2]

            total_rel_cat=len(relation_dict)
            sum=0.1*float(total_rel_cat)/float(rel_oc)
        PathScore.append(math.log(sum)) 
    PathScore=torch.tensor(PathScore,dtype=torch.float32)
    mean=torch.mean(PathScore)
    std=torch.std(PathScore)
    tmp=(PathScore-mean)/std    
    return PathScore
    
if __name__ == "__main__":
    #实体到概念的对照表
    m= numpy.array([0,341, 351, 409, 10980, 11328, 11615, 11807, 11981, 12058, 13166, 13180, 13221, 13223, 13225, 13239, 13270, 14140, 14309, 14493, 14508, 14568, 16929, 16963, 17020, 17035, 17176, 17190, 17207, 17214, 17310, 17316, 25927]  )
    
    file_data = "/yq/yk/huawei/dataset"
    trainfile = file_data + "/conf_train2id.txt"
    # devfile = file_data + "/KBE/datasets/FB15k/test2id.txt"
    testfile = file_data + "/conf_test2id_new.txt"
    folder_path="/yq/yk/huawei/dataset/Path_4/"
    relation_dict={'1': 196, '2': 46292, '3': 55120, '4': 178, '5': 19496, '6': 1782, '7': 1174, '8': 144, '9': 922, '10': 120, '11': 26, '12': 2210, '13': 1248, '14': 14, '15': 50, '16': 2208, '17': 546, '18': 3834, '19': 150, '20': 544, '21': 24, '22': 96, '23': 276, '24': 72, '25': 94, '26': 306, '27': 296, '28': 27022, '29': 700, '30': 502, '31': 1160, '32': 400, '33': 312}

    entity_chain=get_entity_chain(folder_path)
    train_triple, train_confidence = get_data_txt(trainfile)
    test_triple, test_confidence = get_data_txt(testfile)
    PathScore=tf_idf(entity_chain,train_triple,relation_dict)
    print("max:{},min:{},avg:{}".format(np.max(PathScore),np.min(PathScore),np.mean(PathScore)))

# folder_path = "../dataset/Path_4/"
# empty_txt_file_count = 0
# total=0
# for filename in os.listdir(folder_path):
#     total+=1
#     file_path = os.path.join(folder_path, filename)
    
#     if os.path.getsize(file_path) == 0:
#         empty_txt_file_count += 1
# print("文件总数：{}".format(total))
# print(f"文件夹中内容为空的txt文件数量: {empty_txt_file_count}")
