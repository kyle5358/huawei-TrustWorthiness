import pickle
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from calculate import getComplEx,getTransE
from utils import get_data
import pdb
from SuperContrastiveLoss import SupervisedContrastiveLoss
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils_link import get_dict_entityRank,get_data_test

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

# # 搭建神经网络
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

        return F.log_softmax(x, dim=1)[::20]

# 搭建神经网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.gcnh=GCN(100,2)
        self.gcnt=GCN(100,2)
        self.model = nn.Sequential(
            nn.Linear(5, 80),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(80, 10),#输出维度修改
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        # self.model = nn.Sequential(
        #     nn.Linear(7, 80),
        #     nn.ReLU(),
        #     #nn.Dropout(0.5),
        #     nn.Linear(80, 10),#输出维度修改
        #     nn.ReLU(),
        #     nn.Linear(10, 1),
        #     nn.Sigmoid()
        # )
    def forward(self,batch1_edge_index,batch1_x,batch2_edge_index,batch2_x,transE_input,ComplEx_input,path_input,edit_input):
        gcnh_out=self.gcnh(batch1_edge_index,batch1_x)
        gcnt_out=self.gcnt(batch2_edge_index,batch2_x)
        BP_input = torch.cat((transE_input.unsqueeze(1),ComplEx_input.unsqueeze(1),path_input.unsqueeze(1),gcnh_out), dim=1)
        mean = torch.mean(BP_input, dim=0)
        std = torch.std(BP_input, dim=0)+0.0001
        BP_input_normalized = (BP_input - mean) / std
        #BP_input = torch.cat((transE_input.unsqueeze(1),ComplEx_input.unsqueeze(1),gcnh_out), dim=1)
        BP_input_normalized = min_max_normalize(BP_input)
        x = self.model(BP_input_normalized)
        #x = self.model(BP_input)
        return x

def train_model(datafile, modelfile, resultdir, npochos=100, batch_size=50, retrain=False):
    # load training data and test data
    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
        entity2vec, entity2vec_dim, \
        relation2vec, relation2vec_dim, \
        train_triple, train_confidence,train_triple_vec, \
        test_triple, test_confidence, \
        train_transE, test_transE,\
        train_ComplEx,test_ComplEx,\
        train_edit,test_edit,\
        train_path,test_path,\
        train1_datalist,train2_datalist,\
        test1_datalist,test2_datalist= pickle.load(open(datafile, 'rb'))
    
    train_edit=train_edit.to(device)
    test_edit=test_edit.to(device)
    train1_loader = DataLoader(train1_datalist, batch_size=batch_size,shuffle=False)
    train2_loader=DataLoader(train2_datalist, batch_size=batch_size,shuffle=False)
    train3_dataset = TensorDataset(train_triple_vec,train_transE, train_ComplEx, train_path,train_edit,train_confidence)
    train3_loader = DataLoader(train3_dataset, batch_size=batch_size, shuffle=False)

    test1_loader=DataLoader(test1_datalist, batch_size=batch_size,shuffle=False)
    test2_loader=DataLoader(test2_datalist, batch_size=batch_size,shuffle=False)
    test3_dataset = TensorDataset(test_triple,test_transE, test_ComplEx, test_path,test_edit,test_confidence)
    test3_loader = DataLoader(test3_dataset, batch_size=batch_size, shuffle=False)

    # Create the PyTorch model
    nn_model = MLP()
    nn_model=nn_model.to(device)
    # Load pre-trained weights if retrain is True
    if retrain:
        state_dict = torch.load(modelfile)
        nn_model.load_state_dict(state_dict)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    criterion=criterion.to(device)
    #optimizer = optim.Adam(nn_model.parameters(), lr=0.1)
    optimizer = torch.optim.Adam([
    {'params': nn_model.gcnh.parameters(), 'weight_decay': 5e-4},
    {'params': nn_model.gcnt.parameters(), 'weight_decay': 5e-4},
    {'params': nn_model.model.parameters(), 'lr': 0.01}])

    # nn_model.train()

    epoch = 0
    save_inter = 1
    saveepoch = save_inter
    maxF = 0
    earlystopping = 0
    precisions, recalls, f1s, losses = [], [], [], []
    while epoch < npochos:
        epoch += 1
        nn_model.train()
        print('-------training------------:{}'.format(epoch))

        nn_model.train()
        for batch1,batch2,batch3 in zip(train1_loader,train2_loader,train3_loader):
            train_triple_vec,train_transE, train_ComplEx, train_path,train_edit,train_confidence=batch3
    
            optimizer.zero_grad()
            output=nn_model(batch1.edge_index,batch1.x,batch2.edge_index,batch2.x,train_transE, train_ComplEx,train_path,train_edit)
            # print(output)
            # print(train_confidence)
            loss = criterion(output, train_confidence)
            
            # # #[[1],[0],[1]]
            # train_confidence=train_confidence.view(train_confidence.shape[0])
            # criterian=SupervisedContrastiveLoss()
            # criterian=criterian.to(device)
            # new_loss=criterian(train_triple_vec,train_confidence)
            # loss+=new_loss
            loss.backward()
            optimizer.step()

        if epoch >= saveepoch:
            saveepoch += save_inter
            resultfile = os.path.join(resultdir, f"result-{saveepoch}")

            print('-------the test result------------')
            
            [acc, precision, recall, f1score] = test_model(nn_model, test1_loader,test2_loader,test3_loader, resultfile)
            precisions.append(acc)
            losses.append(loss.item())
            if acc > maxF:
                earlystopping = 0
                maxF = acc
                torch.save(nn_model.state_dict(), modelfile)
            else:
                earlystopping += 1
            print(epoch, acc, '  maxF=', maxF,'loss=',loss)

        if earlystopping >= 5:
            break
    plot_metrics(precisions, losses)
    return nn_model

def test_model(model, test1_loader,test2_loader,test3_loader, resultfile):
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    #print("test_confidence:{}".format(test_confidence[0]))
    model.eval()
    with torch.no_grad():
        #EvalScoreFin = open(resultfile + 'EvalScore.txt', 'w')
        for batch1,batch2,batch3 in zip(test1_loader,test2_loader,test3_loader):
                test_triple,test_transE, test_ComplEx, test_path,test_edit,test_confidence=batch3
                outputs=model(batch1.edge_index,batch1.x,batch2.edge_index,batch2.x,test_transE, test_ComplEx,test_path,test_edit)
                for i, res in enumerate(outputs):
                    #tag = np.argmax(res)#[0,1]true,[1,0] false
                    if res >= 0.5:
                        tag = 1
                    else:
                        tag = 0
                    #EvalScoreFin.write(str(test_triple[i][0]) + ' ' + str(test_triple[i][1]) + ' ' + str(test_triple[i][2]) + ' ' + str(test_confidence[i]) + ' ' + str(res) + '\n')

                    #if test_confidence[i][1] == torch.tensor([1]):
                    if test_confidence[i] == torch.tensor([1]):
                        if tag == 1:
                            tp += 1.0#预测正确，预测为正例
                        else:
                            fn += 1.0#预测错误，预测为负例
                    else:
                        if tag == 0:
                            tn += 1.0#预测正确，预测为负例
                        else:
                            fp += 1.0#预测错误，预测为负例
    #EvalScoreFin.close()
    acc = (tp + tn) / float(tp + fn + tn + fp)
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1score = 2 * precision * recall / float(precision + recall)

    return [acc, precision, recall, f1score]

def infer_model(datafile, modelfile, resultfile, batch_size=50):
    # ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
    # entity2vec, entity2vec_dim, \
    # relation2vec, relation2vec_dim, \
    # train_triple, train_confidence,train_triple_vec, \
    # test_triple, test_confidence, \
    # train_transE, test_transE,\
    # train_ComplEx,test_ComplEx,\
    # train_edit,test_edit,\
    # train_path,test_path,\
    # train1_datalist,train2_datalist,\
    # test1_datalist,test2_datalist\
    #     = pickle.load(open(datafile, 'rb'))
    
    # test_edit=test_edit.to(device)
    # test1_loader=DataLoader(test1_datalist, batch_size=batch_size,shuffle=False)
    # test2_loader=DataLoader(test2_datalist, batch_size=batch_size,shuffle=False)
    # test3_dataset = TensorDataset(test_triple,test_transE, test_ComplEx, test_path,test_edit,test_confidence)
    # test3_loader = DataLoader(test3_dataset, batch_size=batch_size, shuffle=False)

    model = MLP()
    state_dict = torch.load(modelfile)
    model.load_state_dict(state_dict)
    model.eval()
    # my_result = test_model(model, test1_loader,test2_loader,test3_loader, resultfile)
    
    # print("acc=" + str(my_result[0]))
    # print("precision={}".format(my_result[1]))
    # print("recall=" + str(my_result[2]))
    # print("f1score=" + str(my_result[3]))
    test_model_linkPrediction(model, datafile)

def get_goldtriples():
    path = "/data1/yk/examine/huawei/TrustWorthiness/dataset/golddataset_new/"
    goldtriples = []

    files = os.listdir(path)
    for file in files:
        # print(file)
        fo = open(path + file, 'r')
        lines = fo.readlines()
        for line in lines:

            nodes = line.rstrip('\n').split(' ')
            goldtriples.append((int(nodes[0]), int(nodes[1]), int(nodes[2])))
        fo.close()
    return goldtriples
from tqdm import tqdm
import pdb
def test_model_linkPrediction(model, datafile):
    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence,train_triple_vec, \
    test_triple_tensor, test_confidence, \
    train_transE, test_transE,\
    train_ComplEx,test_ComplEx,\
    train_edit,test_edit,\
    train_path,test_path,\
    train1_datalist,train2_datalist,\
    test1_datalist,test2_datalist\
        = pickle.load(open(datafile, 'rb'))
    
    #dict_entityRank = get_dict_entityRank(entityRank)
    goldtriples = get_goldtriples()

    totalRawHeadRank = 0.
    totalRawTailRank = 0.
    totalFilterHeadRank = 0.
    totalFilterTailRank = 0.

    totalRawHeadHit10 = 0.
    totalRawTailHit10 = 0.
    totalRawHeadHit1 = 0.
    totalRawTailHit1 = 0.

    totalFilterHeadHit10 = 0.
    totalFilterTailHit10 = 0.
    totalFilterHeadHit1 = 0.
    totalFilterTailHit1 = 0.

    rawTailList = []
    rawHeadList = []
    filterTailList = []
    filterHeadList = []

    test_triple = test_triple_tensor.numpy()
    for i in tqdm(range(0,len(test_triple),2100)):
        print(i)
        rawTailList.clear()
        filterTailList.clear()
        changetriples = []
        for corruptedTailEntity in ent_vocab.values():
            changetriples.append((test_triple[i][0], corruptedTailEntity, test_triple[i][2], 1))

        test_transE,test_ComplEx,test_path,test1_datalist,test2_datalist=get_data_test(ent_vocab, rel_vocab,
             entity2vecfile, relation2vecfile, 
             ComplEx_Entity2vecfile,ComplEx_Relation2vecfile,100,
             concept,relation_dict,path_file,graphfolder_dir,
             changetriples)
        test1_loader=DataLoader(test1_datalist, batch_size=101,shuffle=False)
        test2_loader=DataLoader(test2_datalist, batch_size=101,shuffle=False)
        test3_dataset = TensorDataset(test_transE, test_ComplEx, test_path)
        test3_loader = DataLoader(test3_dataset, batch_size=101, shuffle=False)
        
        results=[]
        for batch1,batch2,batch3 in zip(test1_loader,test2_loader,test3_loader):
                test_transE, test_ComplEx, test_path=batch3
                tmp=model(batch1.edge_index,batch1.x,batch2.edge_index,batch2.x,test_transE, test_ComplEx,test_path,test_edit)
                tmp_np = tmp.squeeze().detach().numpy()
                #pdb.set_trace()
                # 将结果追加到 results 中
                results = np.concatenate((results, tmp_np))
                # results=np.concatenate((results,tmp.detach().numpy()))
        #results=model(test1_loader.edge_index,test1_loader.x,test2_loader.edge_index,test2_loader.x,test_transE, test_ComplEx,test_path)

        # transE = get_TransConfidence(tcthreshold_dict, changetriples, entity2vec, relation2vec)
        # rrank = get_RRankConfidence(rrkthreshold_dict, changetriples, dict_entityRank)

        # results = model.predict([np.array(transE), np.array(rrank)])
        print("len result:{}".format(len(results)))
        for r in range(len(results)):
            rawTailList.append((changetriples[r][1], results[r]))
            if (changetriples[r][0], changetriples[r][1], changetriples[r][2], 1) not in goldtriples:
                filterTailList.append((changetriples[r][1], results[r]))

        rawTailList = sorted(rawTailList, key=lambda sp: sp[1], reverse=True)
        filterTailList = sorted(filterTailList, key=lambda sp: sp[1], reverse=True)
        for j, tri in enumerate(rawTailList):
            # print(tri)
            # print(test_triple[i])
            j = j+1
            if tri[0] == test_triple[i][1]:
                totalRawTailRank += j
                if j <= 10:
                    totalRawTailHit10 +=1.0
                if j == 1:
                    totalRawTailHit1 +=1.0
                break
        for j, tri in enumerate(filterTailList):
            j = j+1
            if tri[0] == test_triple[i][1]:
                totalFilterTailRank += j
                if j <= 10:
                    totalFilterTailHit10 += 1.0
                if j == 1:
                    totalFilterTailHit1 += 1.0
                break
    nums=len(range(0,len(test_triple),2100))
    print("RAW_RANK: ", (totalRawTailRank ) / float(1. * nums))
    print("FILTER_RANK: ", (totalFilterTailRank) / float(1. * nums))
    print("RAW_HIT@10: ", (totalRawTailHit10 ) / float(1. * nums))
    print("FILTER_HIT@10: ", (totalFilterTailHit10) / float(1. * nums))
    print("RAW_HIT@1: ", ( totalRawTailHit1) / float(1. * nums))
    print("FILTER_HIT@1: ", (totalFilterTailHit1) / float(1. * nums))

def plot_metrics(precisions, losses):
    """
    训练指标变化过程可视化
    :param precisions:
    :param recalls:
    :param f1s:
    :param losses:
    :return:
    """
    epochs = range(1, len(precisions) + 1)
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, precisions, 'g', label='Precision')
    # plt.plot(epochs, recalls, 'r', label='Recall')
    # plt.plot(epochs, f1s, 'm', label='F1')
    plt.plot(epochs, losses, 'b', label='Loss')
    plt.title('Training And Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()
    plt.savefig("./precision-loss.png")

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
    graphfolder_dir=file_data + "/subGraphs_1/"
    # entityRank = file_data + "/ResourceRank_4/"
    concept= np.array([0,341, 351, 409, 10980, 11328, 11615, 11807, 11981, 12058, 13166, 13180, 13221, 13223, 13225, 13239, 13270, 14140, 14309, 14493, 14508, 14568, 16929, 16963, 17020, 17035, 17176, 17190, 17207, 17214, 17310, 17316, 25927]  )
    relation_dict={'1': 196, '2': 46292, '3': 55120, '4': 178, '5': 19496, '6': 1782, '7': 1174, '8': 144, '9': 922, '10': 120, '11': 26, '12': 2210, '13': 1248, '14': 14, '15': 50, '16': 2208, '17': 546, '18': 3834, '19': 150, '20': 544, '21': 24, '22': 96, '23': 276, '24': 72, '25': 94, '26': 306, '27': 296, '28': 27022, '29': 700, '30': 502, '31': 1160, '32': 400, '33': 312}

    datafile = "./model/data.pkl"  # 装数据
    modelfile = "./model/model_link.h5"  # 装模型
    resultdir = "../result/"

    batch_size = 64
    retrain = False
    Test = True
    valid = False
    Label = False
    if not os.path.exists(datafile):  # 训练数据若还未加载
        print("Precess data....")
        get_data(entity2idfile, relation2idfile,
                entity2vecfile, relation2vecfile, 
                ComplEx_Entity2vecfile,ComplEx_Relation2vecfile,100,
                concept,relation_dict,path_file,graphfolder_dir,
                trainfile, testfile,
                datafile)
    if not os.path.exists(modelfile):  # 若模型还未训练
        print("data has extisted: " + datafile)
        print("Training model....")
        print(modelfile)
        train_model(datafile, modelfile, resultdir,
                    npochos=200, batch_size=batch_size, retrain=False)
    else:
        if retrain:
            print("ReTraining EE model....")
            train_model(datafile, modelfile, resultdir,
                        npochos=200, batch_size=batch_size, retrain=retrain)

    if Test:
        print("test EE model....")
        print(modelfile)
        infer_model(datafile, modelfile, resultdir, batch_size=batch_size)