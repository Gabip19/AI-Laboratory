import copy
import os
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')


def get_file_path(name):
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'data', 'real', name, name + ".gml")
    return file_path


def read_user_input():
    file_name = input("File name: ")
    comm_num = int(input("Communities number: "))
    return file_name, comm_num


def read_graph_from_file(file_name):
    file_path = get_file_path(file_name)
    graph = nx.read_gml(file_path, label='id')
    return graph


def plot_network(G, communities):
    np.random.seed(333)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(16, 16))
    nx.draw_networkx_nodes(G, pos, node_size=800, cmap=plt.cm.RdYlBu, node_color=communities)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show()


def greedyCommunitiesDetectionByTool(network):
    # Input: a graph
    # Output: list of community index (for every node)

    from networkx.algorithms import community

    A = np.matrix(network["mat"])
    G = nx.from_numpy_matrix(A)
    communities_generator = community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    sorted(map(sorted, top_level_communities))
    communities = [0 for node in range(network['noNodes'])]
    index = 1
    for community in sorted(map(sorted, top_level_communities)):
        for node in community:
            communities[node] = index
        index += 1
    return communities


def most_crossed_edge(G):
    edge_value_pairs = list(nx.edge_betweenness_centrality(G).items())  # toate perechile (muchie, coeficient)
    most_crossed_pair = max(edge_value_pairs, key=lambda item: item[1])  # muchia cea mai traversata cf coeficient
    return most_crossed_pair[0]


def my_greedy_communities_detection(G, no_of_communities=2):
    """
    Determina partiile comunitatilor dintr-un graf dat
    :param G: graful - networkx graph
    :param no_of_communities: numar de comunitati in care dorim sa impartim graful - int
    :return: o lista cu partitiile ale grafului reprezentand comunitatile cerute - list[dict]
    """

    while len(list(nx.connected_components(G))) < no_of_communities:
        # stergem cea mai traversata muchie pana cand avem atatea componente conexe cate comunitati ne dorim
        rez = most_crossed_edge(G)
        source, destination = rez[0], rez[1]
        G.remove_edge(source, destination)

    # construiesc lista de comunitati ca fiind componentele conexe ale grafului
    communities = []
    for comm in nx.connected_components(G):
        communities.append(comm)
    return communities


def start_app():
    file_name, comm_number = read_user_input()
    network = read_graph_from_file(file_name)
    print(network)

    comms = my_greedy_communities_detection(network.copy(), comm_number)

    print("DONE")

    communities_of_node = [0 for _ in range(network.number_of_nodes())]
    for index, comm in enumerate(comms):
        comm = sorted(comm)
        for node in comm:
            communities_of_node[node] = index
        print(comm)
    print(communities_of_node)

    plot_network(network, communities_of_node)


if __name__ == '__main__':
    start_app()
