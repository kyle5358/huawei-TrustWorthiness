
import time
from pygraph.classes.graph import graph  # 修改这里
"""""
生成一阶邻居子图
"""""
def read_all_triples(files):
    graph_dict = {}
    for file_path in files:
        with open(file_path, "r") as file:
            for line in file:
                data = line.split()
                node, neighbor, weight = data[0], data[1], data[2].strip('\n')
                if node not in graph_dict:
                    graph_dict[node] = {neighbor: [weight]}
                    # graph_dict[neighbor] = {node: [weight]}
                else:
                    if neighbor in graph_dict[node]:
                        #graph_dict[neighbor][node].append(weight)
                        graph_dict[node][neighbor].append(weight)
                    else:
                        #graph_dict[neighbor][node] = [weight]
                        graph_dict[node][neighbor] = [weight]
                if neighbor not in graph_dict:
                    graph_dict[neighbor] = {node: [weight]}
                else:
                    if node in graph_dict[neighbor]:
                        graph_dict[neighbor][node].append(weight)
                    else:
                        graph_dict[neighbor][node] = [weight]      
    return graph_dict

def depth_first_search(graph_dict, undirected_graph, node, depth=3):
    depth -= 1
    if depth < 0 or node not in graph_dict:
        return undirected_graph
    sequence = graph_dict[node]
    for key in sequence:
        if not undirected_graph.has_node(key):
            undirected_graph.add_node(key)
        if not undirected_graph.has_edge((node, key)) and not undirected_graph.has_edge((key, node)):
            undirected_graph.add_edge((node, key))  # 修改这里
    for key in sequence:
        undirected_graph = depth_first_search(graph_dict, undirected_graph, key, depth)

    return undirected_graph

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
    file_sub_graphs = file_data + "/subGraphs_1/"
    
    graph_dict = read_all_triples([file_train, file_test])
    print("Graph dictionary size:", len(graph_dict))
    print("ReadAllTriples is done!")

    with open(file_entity, "r") as file:
        for line in file:
            node0 = line.split()[1].strip('\n')
            print("node-----", node0)

            undirected_graph = graph()  # 修改这里
            undirected_graph.add_node(node0)
            t1 = time.perf_counter()
            undirected_graph = depth_first_search(graph_dict, undirected_graph, node0, depth=1)

            with open(file_sub_graphs + node0 + ".txt", "w") as fo:
                node_line = "\t".join(undirected_graph.nodes()) + '\n'
                fo.write(node_line)

                for edge in undirected_graph.edges():
                    edge_line = f"{edge[0]}\t{edge[1]}\n"
                    fo.write(edge_line)

            t2 = time.perf_counter()
            print("Time taken:", t2 - t1)
