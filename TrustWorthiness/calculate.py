import numpy as np
import math
import pdb
import torch
# def getTransE(BatchTriples,entity2vec,relation2vec):
#     TransE_array=[]
#     for triple in BatchTriples:
#         #h+r-t
#         s = entity2vec[triple[0]] + relation2vec[triple[2]] - entity2vec[triple[1]]
#         transV = np.linalg.norm(s, ord=2)
#         # f = 1.0 / (1.0 + math.exp(-1 * (threshold - transV)))
#         # TransE_conf.append(f)
#         TransE_array.append(transV)
#     return TransE_array

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

#def getRotatE():
 
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

def get_Rotate():

    re_score = re_relation * re_tail + im_relation * im_tail
    im_score = re_relation * im_tail - im_relation * re_tail
    re_score = re_score - re_head
    im_score = im_score - im_head

    score = torch.stack([re_score, im_score], dim = 0)
    score = score.norm(dim = 0)

    score = self.gamma.item() - score.sum(dim = 2)