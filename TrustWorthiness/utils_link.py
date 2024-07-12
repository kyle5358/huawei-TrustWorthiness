#nei_graph.py一阶邻居
#SearchPaths2.py
#search.py

from calculate import getComplEx,tcThreshold,get_TransConfidence
import numpy as np
import pickle
import torch
from editConfidence import get_editDistance
from path import get_entity_chain,tf_idf
from GCN import get_gcn
import math
# from TransConfidence import 
def get_dict_entityRank(entityRank):
    dict = {}

    files = os.listdir(entityRank)
    for file in files:
        # print(file)
        fo = open(entityRank + file, 'r')
        lines = fo.readlines()
        dict_l = {}
        for line in lines:

            nodes = line.rstrip('\n').split('\t')
            if nodes[0] == '':
                continue
            dict_l[int(nodes[0])] = float(nodes[1])
        dict[int(os.path.splitext(file)[0])] = dict_l
        fo.close()
    return dict

def getThreshold(rrank):
    distanceFlagList = rrank
    distanceFlagList = sorted(distanceFlagList, key=lambda sp: sp[0], reverse=False)

    threshold = distanceFlagList[0][0] - 0.01
    maxValue = 0
    currentValue = 0
    for i in range(1, len(distanceFlagList)):
        if distanceFlagList[i - 1][1] == 1:
            currentValue += 1
        else:
            currentValue -= 1

        if currentValue > maxValue:
            threshold = (distanceFlagList[i][0] + distanceFlagList[i - 1][0]) / 2.0
            maxValue = currentValue
    # print('threshold... ', threshold)
    return threshold

def tcThreshold(tcDevExamples, entity2vec, relation2vec):
    threshold_dict = {}
    trans_dict = {}

    for tri in tcDevExamples:
        s = entity2vec[tri[0]] + relation2vec[tri[2]] - entity2vec[tri[1]]
        transV = np.linalg.norm(s, ord=2)

        if tri[2] not in trans_dict.keys():
            trans_dict[tri[2]] = [(transV, tri[3])]
        else:
            trans_dict[tri[2]].append((transV, tri[3]))

    for it in trans_dict.keys():
        threshold_dict[it] = getThreshold(trans_dict[it])

    return threshold_dict

def get_TransConfidence(threshold_dict, tcExamples, entity2vec, relation2vec):
    All_conf = 0.0
    confidence_dict = []

    right = 0.0
    for triple in tcExamples:
        if triple[2] in threshold_dict.keys():
            threshold = threshold_dict[triple[2]]
        else:
            threshold = 0.0

        s = entity2vec[triple[0]] + relation2vec[triple[2]] - entity2vec[triple[1]]
        transV = np.linalg.norm(s, ord=2)
        f = 1.0 / (1.0 + math.exp(-1 * (threshold - transV)))
        f = (threshold - transV)

        confidence_dict.append(f)

        if transV <= threshold and triple[3] == 1:
            right += 1.0
            All_conf += f

        elif transV > threshold and triple[3] == -1:
            right += 1.0

    print('TransE-Confidence accuracy ---- ', right / len(tcExamples))
    avg_conf = All_conf / float(len(tcExamples))
    print('avg_confidence ... ', avg_conf, float(len(tcExamples)))
    return confidence_dict
 
def getComplEx(BatchTriples,ent_re_1,ent_im_1,rel_re_1,rel_im_1): # head: (1024,1,2000) relation: (1024,1,2000) tail: (1024,256,2000)
    #对应元素相乘
    ComplEx_array=[]
    for triple in BatchTriples:
        re_head=ent_re_1[triple[0]]
        im_head=ent_im_1[triple[0]]
        re_tail=ent_re_1[triple[1]]
        im_tail=ent_im_1[triple[1]]
        re_relation=rel_re_1[triple[2]]
        im_relation=rel_im_1[triple[2]]

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score + im_head * im_score
        #print(np.shape(score))
        score = score.sum()
        ComplEx_array.append(score)
        
    ComplEx_array=torch.tensor(ComplEx_array,dtype=torch.float32)
    #pdb.set_trace()
    mean=torch.mean(ComplEx_array)
    std=torch.std(ComplEx_array)
    tmp=(ComplEx_array-mean)/std

    return tmp

def get_index(file):
    source_vob = {}
    sourc_idex_word = {}

    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        sourc = line.strip('\r\n').rstrip('\n').split(' ')
        if not source_vob.__contains__(sourc[0]):
            source_vob[sourc[0]] = int(sourc[1])
            sourc_idex_word[int(sourc[1])] = sourc[0]
    f.close()
    return source_vob, sourc_idex_word

def load_transE_vec(fname, vocab, k=300):
    f = open(fname)
    w2v={}
    W = np.zeros(shape=(vocab.__len__() + 2, k))
    unknowtoken = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["**UNK**"]
            unknowtoken +=1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[word]

    # print('!!!!!! UnKnown tokens in w2v', unknowtoken)
    # for sss in W:
    #     print(sss)
    return k, W

# def load_RotatE_vec(fname,vocab,k=100):
def load_ComplEx_vec(fname,vocab,k=100):
    f = open(fname)
    r2v={}
    i2v={}
    re_1 = np.zeros(shape=(vocab.__len__() + 2, k))
    im_1 = np.zeros(shape=(vocab.__len__() + 2, k))   
    unknowtoken = 0
    for line in f:
        values = line.split(',')
        word = values[0].strip()
        #print(word+"hh")
        retemp=values[1].strip().split()
        re = np.asarray([float(part) for part in retemp], dtype='float32')
        imtemp=values[2].strip().split()
        im = np.asarray([float(part) for part in imtemp], dtype='float32')
        # print("im:{}".format(im))    
        r2v[word] = re
        i2v[word] = im
    f.close()
    i2v["**UNK**"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not i2v.__contains__(word):
            i2v[word] = i2v["**UNK**"]
            unknowtoken +=1
            re_1[vocab[word]] = r2v[word]
            im_1[vocab[word]] = i2v[word]
        else:
            re_1[vocab[word]] = r2v[word]
            im_1[vocab[word]] = i2v[word]

    # print('!!!!!! UnKnown tokens in w2v', unknowtoken)
    # for sss in W:
    #     print(sss)
    return k, re_1,im_1
  
# def get_data_txt(trainfile):
#     train_triple = []
#     train_confidence = []

#     f = open(trainfile, "r")
#     lines = f.readlines()
#     for line in lines:
#         tri = line.rstrip('\r\n').rstrip('\n').rstrip('\r').split(' ')
#         train_triple.append((int(tri[0]), int(tri[1]), int(tri[2]), int(tri[3])))
#         if tri[3] == '1':
#             train_confidence.append([1])
#         else:
#             train_confidence.append([0])

#     f.close()
#     return train_triple, train_confidence

def get_data_test(ent_vocab, rel_vocab,
             entity2vecfile, relation2vecfile, 
             ComplEx_Entity2vecfile,ComplEx_Relation2vecfile,w2v_k,
             concept,relation_dict,path_file,graphfolder_dir,
             test_triple):

    # ent_vocab, ent_idex_word = get_index(entity2idfile)
    # rel_vocab, rel_idex_word = get_index(relation2idfile)   

    entvec_k, entity2vec = load_transE_vec(entity2vecfile, ent_vocab, k=w2v_k)
    relvec_k, relation2vec = load_transE_vec(relation2vecfile, rel_vocab, k=w2v_k)
    #print("entity2vec  size:{}, relation2vec size:{}".format(len(entity2vec),len(relation2vec)))

    # #tcthreshold_dict = tcThreshold(train_triple, entity2vec, relation2vec)
    # train_transE = getTransE( train_triple, entity2vec, relation2vec)
    # test_transE = getTransE( test_triple, entity2vec, relation2vec)
    # print("train_transE:avg:{}".format(np.mean(train_transE)))
    # print("test_transE:avg:{}".format(np.mean(test_transE)))
    # train_transE = torch.tensor(train_transE, dtype=torch.float32)
    # test_transE = torch.tensor(test_transE, dtype=torch.float32)
    
    #ComplEx的分数
    #load_ComplEx_vec(ComplEx_Relation2vecfile, ent_vocab, k=w2v_k)
    _, ent_re_1,ent_im_1= load_ComplEx_vec(ComplEx_Entity2vecfile, ent_vocab, k=100)
    _,rel_re_1,rel_im_1=load_ComplEx_vec(ComplEx_Relation2vecfile, rel_vocab, k=100)
    # print(ent_im_1)
    #print("entity2vec  size:{}， relation2vec size:{}".format(np.shape(ent_re_1),np.shape(rel_re_1)))

    # train_ComplEx = getComplEx( train_triple, ent_re_1,ent_im_1,rel_re_1,rel_im_1)
    test_ComplEx = getComplEx( test_triple, ent_re_1,ent_im_1,rel_re_1,rel_im_1)
    # train_ComplEx = torch.tensor(train_ComplEx, dtype=torch.float32)
    test_ComplEx = torch.tensor(test_ComplEx, dtype=torch.float32)
    # print("train_ComplEx:avg:{}".format(torch.mean(train_ComplEx)))
    #print("test_ComplEx:avg:{}".format(torch.mean(test_ComplEx)))  

    #TransE的分数
    tcthreshold_dict_transE = tcThreshold(test_triple, entity2vec, relation2vec)
    #train_transE = get_TransConfidence(tcthreshold_dict_transE, train_triple, entity2vec, relation2vec)
    test_transE = get_TransConfidence(tcthreshold_dict_transE, test_triple, entity2vec, relation2vec)
    test_transE = torch.tensor(test_transE, dtype=torch.float32)
    #print("train_transE:avg:{}".format(torch.mean(train_transE)))
    #print("test_transE:avg:{}".format(torch.mean(test_transE)))

    # #editdistance分数
    # # train_edit=get_editDistance(train_triple)
    # test_edit=get_editDistance(test_triple)
    # # train_edit = torch.tensor(train_edit, dtype=torch.float32).cuda()
    # test_edit= torch.tensor(test_edit, dtype=torch.float32).cuda()
    
    #path分数
    folder_path=path_file
    entity_chain=get_entity_chain(path_file)
    # train_path=tf_idf(entity_chain,train_triple,relation_dict,folder_path)
    test_path=tf_idf(entity_chain,test_triple,relation_dict,folder_path)
    # train_path = torch.tensor(train_path, dtype=torch.float32)
    test_path = torch.tensor(test_path, dtype=torch.float32)

    #GCN embedding
    print("get gcn embedding")
    # train1_datalist,train2_datalist=get_gcn(train_triple,graphfolder_dir,entity2vec)
    test1_datalist,test2_datalist=get_gcn(test_triple,graphfolder_dir,entity2vec)

    # out = open(datafile,'wb')
    # pickle.dump([ent_vocab, ent_idex_word, rel_vocab, rel_idex_word,
    #              entity2vec, entvec_k,
    #              relation2vec, relvec_k,
    #              train_triple, train_confidence,train_triple_vec,
    #              test_triple, test_confidence,
    #              train_transE, test_transE,
    #              train_ComplEx,test_ComplEx,
    #              train_edit,test_edit,
    #              train_path,test_path,
    #              train1_datalist,train2_datalist,
    #              test1_datalist,test2_datalist
    #               ], out)
    # out.close()
    print ("dataset created!")
    return [test_transE,test_ComplEx,test_path,test1_datalist,test2_datalist]

if __name__ == "__main__":
    file_data = "/data1/yk/examine/huawei/TrustWorthiness/dataset"
    entity2idfile = file_data + "/entity2id.txt"
    relation2idfile = file_data + "/relation2id.txt"

    entity2vecfile = file_data + "/Entity2vec.txt"
    relation2vecfile = file_data + "/Relation2vec.txt"
    ComplEx_Relation2vecfile=file_data + "/ComplEx_Relation2vec.txt"
    ComplEx_Entity2vecfile=file_data + "/ComplEx_Entity2vec.txt"
    trainfile = file_data + "/conf_train2id.txt"
    # devfile = file_data + "/KBE/datasets/FB15k/test2id.txt"
    testfile = file_data + "/conf_test2id_new.txt"  
    path_file = file_data + "/Path_4/"
    # entityRank = file_data + "/ResourceRank_4/"
    graphfolder_dir=file_data + "/subGraphs_1/"
    # folder_path="/yq/yk/huawei/dataset/Path_4/"

    concept= np.array([0,341, 351, 409, 10980, 11328, 11615, 11807, 11981, 12058, 13166, 13180, 13221, 13223, 13225, 13239, 13270, 14140, 14309, 14493, 14508, 14568, 16929, 16963, 17020, 17035, 17176, 17190, 17207, 17214, 17310, 17316, 25927]  )
    relation_dict={'1': 196, '2': 46292, '3': 55120, '4': 178, '5': 19496, '6': 1782, '7': 1174, '8': 144, '9': 922, '10': 120, '11': 26, '12': 2210, '13': 1248, '14': 14, '15': 50, '16': 2208, '17': 546, '18': 3834, '19': 150, '20': 544, '21': 24, '22': 96, '23': 276, '24': 72, '25': 94, '26': 306, '27': 296, '28': 27022, '29': 700, '30': 502, '31': 1160, '32': 400, '33': 312}

    datafile = "./model/data_new.pkl"  # 装数据
    modelfile = "./model/model.h5"  # 装模型
    resultdir = "./result/"
    resultdir = "./result/Model---"
