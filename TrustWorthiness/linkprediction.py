import torch
import os
from util.test_util import *
from tqdm import tqdm
import torch.nn.functional as F
from config import config
from util.parameter_util import load_o_emb, load_wb

def _calc(h, t, r, norm):
    return torch.norm(h + r - t, p=norm, dim=1).cpu().numpy().tolist()

def predict(batch, entity_emb, relation_emb, norm):
    pos_hs = batch[:, 0]
    pos_rs = batch[:, 1]
    pos_ts = batch[:, 2]

    pos_hs = torch.IntTensor(pos_hs).cuda()
    pos_rs = torch.IntTensor(pos_rs).cuda()
    pos_ts = torch.IntTensor(pos_ts).cuda()

    p_score = _calc(entity_emb[pos_hs.type(torch.long)],
                    entity_emb[pos_ts.type(torch.long)],
                    relation_emb[pos_rs.type(torch.long)],
                    norm)
    return p_score

def predict_confidence(batch, entity_emb, relation_emb, norm, w, b):
    h = torch.IntTensor(batch[:, 0]).cuda()
    r = torch.IntTensor(batch[:, 1]).cuda()
    t = torch.IntTensor(batch[:, 2]).cuda()
    score = _calc(entity_emb[h.type(torch.long)], entity_emb[t.type(torch.long)], relation_emb[r.type(torch.long)], norm)
    conf = torch.sigmoid(w * torch.tensor(score).cuda() + b)
    return conf


def test_head(golden_triple, train_set, entity_emb, relation_emb, norm):
    head_batch = get_head_batch(golden_triple, len(entity_emb))  # 头实体被随机替换，的三元组集合
    value = predict(head_batch, entity_emb, relation_emb, norm)  # 每一个三元组的得分
    golden_value = value[golden_triple[0]]  # 正确三元组的得分
    # li = np.argsort(value)
    res = 1
    sub = 0

    for pos, val in enumerate(value):
        if val < golden_value:
            res += 1
            if (pos, golden_triple[1], golden_triple[2]) in train_set:
                sub += 1

            
    return res, res - sub, bad_cases


def test_tail(golden_triple, train_set, entity_emb, relation_emb, norm):
    tail_batch = get_tail_batch(golden_triple, len(entity_emb))
    value = predict(tail_batch, entity_emb, relation_emb, norm)
    golden_value = value[golden_triple[2]]
    # li = np.argsort(value)
    res = 1
    sub = 0
    bad_cases = []
    for pos, val in enumerate(value):
        if val < golden_value:
            res += 1
            if (golden_triple[0], golden_triple[1], pos) in train_set:
                sub += 1

    return res, res - sub, bad_cases

# def test_conf_prediction(test_list, entity_emb, relation_emb, norm,  w, b):
#     test_total = len(test_list)
#     batch = np.zeros((test_total, 3), dtype=np.int32)
#     golden_conf = np.zeros(test_total, dtype=np.float32)
#     for i, test_triple in enumerate(test_list):
#         batch[i, 0] = test_triple[0]
#         batch[i, 1] = test_triple[1]
#         batch[i, 2] = test_triple[2]
#         golden_conf[i] = test_triple[3]
#     pred_conf = predict_confidence(batch, entity_emb, relation_emb, norm, w, b)
#     golden_conf = torch.FloatTensor(golden_conf).cuda()
#     mse = torch.sum((pred_conf - golden_conf) ** 2) / test_total
#     print('\t\t\tConfidence Prediction\t\t\t')
#     print('MSE\t\t\t%f\t\t\t' % mse)
#     return mse.item()

def test_link_prediction(test_list, train_set, entity_emb, relation_emb, norm):
    test_total = len(test_list)

    l_hit1 = 0
    r_hit1 = 0
    l_hit3 = 0
    r_hit3 = 0
    l_hit10 = 0
    r_hit10 = 0

    l_hit1_filter = 0
    r_hit1_filter = 0
    l_hit3_filter = 0
    r_hit3_filter = 0
    l_hit10_filter = 0
    r_hit10_filter = 0

    l_mr = 0
    r_mr = 0

    l_mr_filter = 0
    r_mr_filter = 0

    l_mrr = 0
    r_mrr = 0

    l_mrr_filter = 0
    r_mrr_filter = 0

    head_bad_cases = []
    tail_bad_cases = []

    for golden_triple in tqdm(test_list, desc='Processing Link Prediction'):
        # print('test ---' + str(i) + '--- triple')
        # print(i, end="\r")
        l_pos, l_filter_pos, head_bad_case = test_head(golden_triple, train_set, entity_emb, relation_emb, norm)
        r_pos, r_filter_pos, tail_bad_case = test_tail(golden_triple, train_set, entity_emb, relation_emb, norm)  # position, 1-based

        head_bad_cases.extend(head_bad_case)  # 收集头实体的bad cases
        tail_bad_cases.extend(tail_bad_case)  # 收集尾实体的bad cases
        
        # print(golden_triple, end=': ')
        # print('l_pos=' + str(l_pos), end=', ')
        # print('l_filter_pos=' + str(l_filter_pos), end=', ')
        # print('r_pos=' + str(r_pos), end=', ')
        # print('r_filter_pos=' + str(r_filter_pos), end='\n')


        if l_pos <= 10:
            l_hit10 += 1
        if r_pos <= 10:
            r_hit10 += 1
        if l_filter_pos <= 10:
            l_hit10_filter += 1
        if r_filter_pos <= 10:
            r_hit10_filter += 1

        if l_pos <= 3:
            l_hit3 += 1
        if r_pos <= 3:
            r_hit3 += 1
        if l_filter_pos <= 3:
            l_hit3_filter += 1
        if r_filter_pos <= 3:
            r_hit3_filter += 1

        if l_pos <= 1:
            l_hit1 += 1
        if r_pos <= 1:
            r_hit1 += 1
        if l_filter_pos <= 1:
            l_hit1_filter += 1
        if r_filter_pos <= 1:
            r_hit1_filter += 1

        l_mr += l_pos
        r_mr += r_pos

        l_mr_filter += l_filter_pos
        r_mr_filter += r_filter_pos

        l_mrr += 1/l_pos
        r_mrr += 1/r_pos

        l_mrr_filter += 1/l_filter_pos
        r_mrr_filter += 1/r_filter_pos

    l_mr /= test_total
    r_mr /= test_total

    l_mr_filter /= test_total
    r_mr_filter /= test_total

    l_mrr /= test_total
    r_mrr /= test_total

    l_mrr_filter /= test_total
    r_mrr_filter /= test_total

    l_hit10 /= test_total
    r_hit10 /= test_total

    l_hit10_filter /= test_total
    r_hit10_filter /= test_total

    l_hit3 /= test_total
    r_hit3 /= test_total

    l_hit3_filter /= test_total
    r_hit3_filter /= test_total

    l_hit1 /= test_total
    r_hit1 /= test_total

    l_hit1_filter /= test_total
    r_hit1_filter /= test_total

    print('\t\t\tmean_rank\t\t\t')
    print('head(raw)\t\t\t%.3f\t\t\t' % l_mr)
    print('tail(raw)\t\t\t%.3f\t\t\t' % r_mr)
    print('average(raw)\t\t\t%.3f\t\t\t' % ((l_mr + r_mr) / 2))

    print('head(filter)\t\t\t%.3f\t\t\t' % l_mr_filter)
    print('tail(filter)\t\t\t%.3f\t\t\t' % r_mr_filter)
    print('average(filter)\t\t\t%.3f\t\t\t' % ((l_mr_filter + r_mr_filter) / 2))

    print('\t\t\tMRR\t\t\t')
    print('head(raw)\t\t\t%.3f\t\t\t' % l_mrr)
    print('tail(raw)\t\t\t%.3f\t\t\t' % r_mrr)
    print('average(raw)\t\t\t%.3f\t\t\t' % ((l_mrr + r_mrr) / 2))

    print('head(filter)\t\t\t%.3f\t\t\t' % l_mrr_filter)
    print('tail(filter)\t\t\t%.3f\t\t\t' % r_mrr_filter)
    print('average(filter)\t\t\t%.3f\t\t\t' % ((l_mrr_filter + r_mrr_filter) / 2))

    print('\t\t\thit@10\t\t\t')
    print('head(raw)\t\t\t%.3f\t\t\t' % l_hit10)
    print('tail(raw)\t\t\t%.3f\t\t\t' % r_hit10)
    print('average(raw)\t\t\t%.3f\t\t\t' % ((l_hit10 + r_hit10) / 2))

    print('head(filter)\t\t\t%.3f\t\t\t' % l_hit10_filter)
    print('tail(filter)\t\t\t%.3f\t\t\t' % r_hit10_filter)
    print('average(filter)\t\t\t%.3f\t\t\t' % ((l_hit10_filter + r_hit10_filter) / 2))

    print('\t\t\thit@3\t\t\t')
    print('head(raw)\t\t\t%.3f\t\t\t' % l_hit3)
    print('tail(raw)\t\t\t%.3f\t\t\t' % r_hit3)
    print('average(raw)\t\t\t%.3f\t\t\t' % ((l_hit3 + r_hit3) / 2))

    print('head(filter)\t\t\t%.3f\t\t\t' % l_hit3_filter)
    print('tail(filter)\t\t\t%.3f\t\t\t' % r_hit3_filter)
    print('average(filter)\t\t\t%.3f\t\t\t' % ((l_hit3_filter + r_hit3_filter) / 2))

    print('\t\t\thit@1\t\t\t')
    print('head(raw)\t\t\t%.3f\t\t\t' % l_hit1)
    print('tail(raw)\t\t\t%.3f\t\t\t' % r_hit1)
    print('average(raw)\t\t\t%.3f\t\t\t' % ((l_hit1 + r_hit1) / 2))

    print('head(filter)\t\t\t%.3f\t\t\t' % l_hit1_filter)
    print('tail(filter)\t\t\t%.3f\t\t\t' % r_hit1_filter)
    print('average(filter)\t\t\t%.3f\t\t\t' % ((l_hit1_filter + r_hit1_filter) / 2))
    # return l_hit10_filter, r_hit10_filter

    # print("Head Bad Cases:")
    # for bad_case in head_bad_cases:
    #     print(bad_case)
    # print("/n")  
    # print("Tail Bad Cases:")
    # for bad_case in tail_bad_cases:
    #     print(bad_case)
    

    return (l_hit10_filter + r_hit10_filter) / 2, (l_hit10 + r_hit10) / 2

if __name__ == "__main__":
    entity_emb, relation_emb = load_o_emb(config.res_dir, config.entity_total, config.relation_total, config.dim)
    w, b = load_wb(config.res_dir)
    print('test link prediction starting...')
    train_list = [i[:3] for i in config.train_list]
    test_link_prediction(config.test_list, set(train_list), entity_emb, relation_emb, config.norm)
    #test_conf_prediction(config.test_list, entity_emb, relation_emb, config.norm, w, b)
    print('test link prediction ending...')
