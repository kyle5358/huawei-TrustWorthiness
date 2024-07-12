#用的是golddata生成的子图
#coding = utf-8
#生成子图subgraph
import time
from pygraph.classes.digraph import digraph
#import digraph
import os

def ReadAllTriples(files):
    dict = {}

    for f in files:
        file = open(f, "r")
        for line in file:
            list = line.split(" ")

            if list[0] in dict.keys():
                if list[1] in dict.get(list[0]).keys():
                    dict.get(list[0]).get(list[1]).append(list[2].strip('\n'))
                else:
                    dict.get(list[0])[list[1]] = [list[2].strip('\n')]
            else:
                dict[list[0]] = {list[1]:[list[2].strip('\n')]}

        # for key in dict.keys():
        #     print(key+' : ',dict[k])
        file.close()

    return dict

def DFS(dict, dg, node, depth=3):
    depth -= 1

    if depth < 0:
        return dg
    if node not in dict.keys():
        return dg
    sequence = dict[node]
    count = 0
    for key in sequence.keys():
        if not dg.has_node(key):
            dg.add_node(key)
        if not dg.has_edge((node, key)):
            dg.add_edge((node, key), wt=len(sequence[key]))
            count += len(sequence[key])
        else:
            continue
            # print(node, key, dg.edge_weight((node, key)), len(sequence[key]))

        # array[int(node)][int(key)] = len(sequence[key])
        dg = DFS(dict, dg, key, depth)

    for n in dg.neighbors(node):
        dg.set_edge_weight((node, n),wt= float(dg.edge_weight((node, n))/max(count,1)))

    return dg

if __name__ == '__main__':

    file_data = "/data1/yk/examine/huawei/TrustWorthiness/dataset"
    file_entity = file_data + "/entity2id.txt"
    # file_train = file_data + "/train.txt"
    # file_test = file_data + "/test.txt"
    # file_valid = file_data + "/valid.txt"
    file_train = file_data + "/train2id.txt"
    file_test = file_data + "/test2id_new.txt"
    # file_valid = file_data + "/conf_valid2id.txt"
    # file_subGraphs = file_data + "/subGraphs_4/"
    file_subGraphs = file_data + "/2part_subGraphs_1/"
    
    dict = ReadAllTriples([file_train, file_test])
    print("dict size--", dict.__len__())
    print("ReadAllTriples is done!")

    file = open(file_entity, "r")

    for line in file:
        list = line.split(" ")
        node0 = list[1].strip('\n')
        print("node0-----", node0)

        dg = digraph()
        dg.add_node(node0)
        t1 = time.perf_counter()
        # dg = DFS(dict, dg, node0, depth=4)
        dg = DFS(dict, dg, node0, depth=1)
        fo = open(file_subGraphs + node0 + ".txt", "w")
        NODE = ""
        for nodei in dg.nodes():
            NODE = NODE +nodei+ "\t"
        fo.write(NODE+'\n')

        for e in dg.edges():
            fo.write(e[0] + "\t" + e[1] + "\t" + str(dg.edge_weight(e))+'\n')
        fo.close()

        t2=time.perf_counter()
        # time.sleep(1)
        print(t2-t1)
        # print(dg.nodes().__len__())
        # for edge in dg.edges():
        #     print('edge----',edge)
    file.close()






