import copy
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')

'''
JAZZ NETWORK of Gleiser and Danon
Physics E-print Archive at arxiv.org if they have coauthored one or more papers posted on the archive.
'''


def greedy_community_detection1(G):
    # Start with each node in its own community
    communities = [{n} for n in G.nodes()]

    # Compute the modularity matrix
    A = nx.adjacency_matrix(G)
    degrees = dict(G.degree())
    m = G.number_of_edges()
    B = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    for i in range(G.number_of_nodes()):
        for j in range(G.number_of_nodes()):
            if i == j:
                B[i, j] = 0
            else:
                B[i, j] = A[i, j] - degrees[i] * degrees[j] / (2 * m)

    iteration = 0
    # Repeat until no further improvement can be achieved
    while True:
        # Compute the increase in modularity for each node
        deltas = {}
        for node in G.nodes():
            current_community = [c for c in communities if node in c][0]
            current_modularity = sum(B[i, j] for i in current_community for j in current_community)
            new_modularities = []
            for neighbor in G.neighbors(node):
                neighbor_community = [c for c in communities if neighbor in c][0]
                new_community = current_community.union(neighbor_community)
                new_modularity = sum(B[i, j] for i in new_community for j in new_community)
                delta_modularity = new_modularity - current_modularity
                new_modularities.append((new_community, delta_modularity))
            best_community, best_delta = max(new_modularities, key=lambda x: x[1])
            deltas[node] = (best_community, best_delta)

        # Select the node that results in the largest increase in modularity
        best_node, best_data = max(deltas.items(), key=lambda x: x[1][1])

        iteration += 1
        # If the best node does not result in a positive increase in modularity, stop
        if best_data[1] <= 0 or iteration == 100:
            break

        # Move the best node to its new community
        old_community = [c for c in communities if best_node in c][0]
        new_community = best_data[0]
        communities.remove(old_community)
        communities.append(new_community)

    return communities


def greedy_communities_detection_by_tool(network):
    # Input: a graph
    # Output: list of comunity index (for every node)

    from networkx.algorithms import community

    communities_generator = community.girvan_newman(network)
    top_level_communities = next(communities_generator)
    sorted(map(sorted, top_level_communities))
    communities = [0 for node in network.nodes]
    index = 1
    for community in sorted(map(sorted, top_level_communities)):
        for node in community:
            communities[node] = index
        index += 1
    return communities


def plot_network(network, communities):
    np.random.seed(333)  # to freeze the graph's view (networks uses a random view)
    pos = nx.spring_layout(network)  # compute graph layout
    plt.figure(figsize=(16, 16))  # image is 8 x 8 inches
    nx.draw_networkx_nodes(network, pos, node_size=600, cmap=plt.cm.RdYlBu, node_color=communities)
    nx.draw_networkx_edges(network, pos, alpha=0.3)
    plt.show()


def get_file_path(name):
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'data', 'real', name, name + ".gml")
    print(file_path)
    return file_path


def read_graph_from_file(file_name):
    file_path = get_file_path(file_name)
    graph = nx.read_gml(file_path, label='id')
    return graph


"""
/////////////////////////////// MERGE SA IMPARTA IN DOUA COMUNITATI MAJORITARE /////////////////////////////////////////
"""

def calcul_aij(G, c_i, C):
    rez = sum([calcul_eij(G, c_i, c_j) for c_j in C if c_j != c_i])
    return rez

def calcul_eij(G, c_i, c_j):
    eij = 0
    for node_i in c_i:
        for node_j in c_j:
            if G.has_edge(node_i, node_j):
                eij += 1
    return eij

def clauset_newman_moore(G, comm_needed=1000):
    # Initialize each node to its own community
    C = [{i} for i in G.nodes()]
    # Compute the total number of edges in G
    m = G.number_of_edges()

    # Repeat until no further improvement in modularity can be achieved
    while True:
        # Compute the change in modularity ΔQ for each pair of communities
        # ∆Q = eij + eji − 2aiaj = 2(eij − aiaj)
        delta_Q = {}
        for i, c_i in enumerate(C):
            for j, c_j in enumerate(C):
                if i >= j:
                    continue
                e_ij = calcul_eij(G, c_i, c_j)
                k_i = sum([G.degree(n) for n in c_i])
                k_j = sum([G.degree(n) for n in c_j])
                delta_Q[(i, j)] = e_ij / m - (k_i * k_j) / (4 * m * m)

        # If no further increase in modularity is possible, terminate and return the communities
        if not delta_Q or len(C) == comm_needed:
            return C

        # Merge the pair of communities that results in the greatest increase in modularity delta Q
        i, j = max(delta_Q, key=delta_Q.get)
        C[i] = C[i].union(C[j])
        del C[j]
    return C

"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

def are_connected(G, c_i, c_j):
    for node_1 in c_i:
        for node_2 in c_j:
            if G.has_edge(node_1, node_2):
                return True
    return False

def greedy_comms(G, comms_num=None):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    adj_mat = nx.to_numpy_array(G)

    # max_q = -1 * np.inf
    comms = [{i} for i in G.nodes()]

    while True:
        # Q = 0.0
        edge_mat = np.zeros((len(comms), len(comms)))
        a_mat = np.zeros(len(comms))

        for i, c_i in enumerate(comms):
            for j, c_j in enumerate(comms):
                if i < j:
                    edge_mat[i][j] = calcul_eij(G, c_i, c_j)
                    edge_mat[j][i] = edge_mat[i][j]
        # print(edge_mat)

        for i, c_i in enumerate(comms):
            a_mat[i] = sum(edge_mat[i][j] for j in range(len(comms)) if i != j)

        # for i, c in enumerate(comms):
        #     Q += edge_mat[i][i] - sum(edge_mat[i]) * sum(edge_mat[i])

        comm_pair = None
        max_delta_q = -1 * np.inf

        for i, c_i in enumerate(comms):
            for j, c_j in enumerate(comms):
                if i < j and are_connected(G, c_i, c_j):
                    delta_q = edge_mat[i][j] + edge_mat[j][i] - 2 * a_mat[i] * a_mat[j]
                    if delta_q > max_delta_q:
                        max_delta_q = delta_q
                        comm_pair = (i, j)
        print(max_delta_q)
        if comm_pair is not None:
            comm_A = comm_pair[0]
            comm_B = comm_pair[1]
            comms[comm_A] = comms[comm_A].union(comms[comm_B])
            del comms[comm_B]

            # Q += max_delta_q
            # if max_q < Q:
            #     max_q = Q
            if comms_num is not None and len(comms) <= comms_num:
                return comms
        else:
            return comms

    return comms

"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""

def last_chance(G, comms_num=None):
    C = [{i} for i in G.nodes]
    old_q = nx.algorithms.community.modularity(G, C)

    while True:
        max_q = -np.inf
        good_pair = None
        for i, com_i in enumerate(C):
            for j, com_j in enumerate(C):
                if i < j:
                    current_comm = copy.deepcopy(C)
                    current_comm[i] = current_comm[i].union(current_comm[j])
                    del current_comm[j]
                    current_q = nx.algorithms.community.modularity(G, current_comm)

                    if current_q > max_q:
                        max_q = current_q
                        good_pair = current_comm

                    print(current_q, max_q)
        if comms_num is not None:
            if comms_num == len(C):
                break
        elif old_q >= max_q:
            break
        if good_pair is not None:
            old_q = max_q
            C = good_pair
    return C

"""
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


class Configuration:
    def _init_(self, e, a, Q0, communities):
        self.e = e
        self.a = a
        self.Q0 = Q0
        self.communities = communities

    def join(self, e, i, j):
        """
        Joins two communities
        :param e: the matrix of connections between different communities
        :param i: one end (a community) of the edge that connects two communities
        :param j: the end (a community) of the edge that connects two communities
        :return: the matrix of connections after join was made
        """
        e[i][i] += e[j][j] + 2 * e[i][j]
        e[i][j] = 0
        e[j][i] = 0
        e[j][j] = 0
        for k in range(network["noNodes"]):
            if k == i or k == j:
                continue
            e[i][k] += e[j][k]
            e[k][i] += e[k][j]
            e[j][k] = 0
            e[k][j] = 0
        return e

    def joinCommunities(self):
        """
        Finds the edge that maximizes the difference between the modularity of the old
        matrix of connections and the new one, obtained after simulating the morphing
        of the communities that correspond to the ends of that edge, and applies it,
        modifying e, a and Q0
        """
        maximumDeltaQ = -np.inf
        e_maximumSimulation = []
        a_maximumSimulation = []
        i = network["noNodes"]-1
        (u, v) = (0, 0)
        while i > -1:
            j = network["noNodes"]-1
            while j > -1:
                if self.e[i][j] != 0 and i != j:
                    deltaQ = 2 * (2*network["noEdges"]*self.e[i][j] - self.a[i] * self.a[j])
                    # print("Candidate = ", i, j, f"--> deltaQ = {deltaQ}")
                    if deltaQ > maximumDeltaQ:
                        maximumDeltaQ = deltaQ
                        e_maximumSimulation = copy_mat(self.join(copy_mat(self.e), i, j))
                        a_maximumSimulation = []
                        for k in range(network["noNodes"]):
                            a_maximumSimulation.append(sum(e_maximumSimulation[k]))
                        (u, v) = (i, j)
                j -= 1
            i -= 1
        # print("Maxim = ", u, v)
        # print("Bag comunitatea ", v, " in ", u)
        for c in range(network["noNodes"]):
            if self.communities[c] == v:
                self.communities[c] = u
        self.Q0 = self.Q0 + maximumDeltaQ
        self.e = e_maximumSimulation
        self.a = a_maximumSimulation


def reachedFinality(e):
    """
    Helps to find out if the graph has reached finality, meaning the point where it contains only one community
    :param e: the matrix of connections between different communities
    :return: 0 if finality was not reached, 1 if it was
    """
    nr = 0
    for i in range(network["noNodes"]):
        for j in range(network["noNodes"]):
            if e[i][j] != 0:
                nr += 1
    if nr == 1:
        return 1
    else:
        return 0


def greedyCommunitiesDetection(network, nr):
    """
    Detects the communities and their vertexes
    :param network: the network in which we try to form nr communities
    :param nr: the number of communities we want to form
    :return: dictionary of communities and the vertexes they contain
    """
    e = copy_mat(network["mat"])
    # print_e(e)
    a = network["degrees"].copy()
    # print_a(a)
    Q0 = 0
    communities = []
    for i in range(network["noNodes"]):
        Q0 = Q0 + network["noEdges"]*e[i][i] - a[i]*a[i]
        communities.append(i)
    # print(Q0)
    # print(f"Communities:.... {communities}")
    c = Configuration(e, a, Q0, communities)

    c.joinCommunities()
    result = {}
    while reachedFinality(e) == 0 and len(np.unique(communities)) != nr:
        e = copy.deepcopy(c.e)
        # print_e(e)
        a = c.a.copy()
        # print_a(a)
        Q0 = c.Q0
        # print(Q0)
        communities = c.communities.copy()
        dicti = {}
        for i in range(len(communities)):
            if not (communities[i] in dicti):
                dicti[communities[i]] = []
            dicti[communities[i]] += [i]
        if len(dicti.keys()) == nr:
            result = communities
            print(f"Comunitati:...{dicti}")
            print(f"Nr comunitati = {len(dicti.keys())}")
            print(communities)
            break
        c.joinCommunities()
    return result


def merge_perfect(network):
    n = network.number_of_nodes()
    m = network.number_of_edges()
    e = copy.deepcopy(nx.to_numpy_array(network))
    a = [network.degree(i) for i in network.nodes]
    print("aaaa")

    Q0 = 0
    communities = []
    for i in network.nodes:
        Q0 = Q0 + m * e[i][i] - a[i] * a[i]
        communities.append({i})

    print(Q0)


def start_app():
    network = read_graph_from_file("dolphins")
    print(network)
    # comms = last_chance(network)
    comms = merge_perfect(network)
    communities = [0 for _ in range(network.number_of_nodes())]
    for comm in comms:
        print(sorted(comm))
    for index, comm in enumerate(comms):
        for node in comm:
            communities[node] = index
    print("Done")
    # print(communities)
    # del communities[0]
    plot_network(network, communities)


if __name__ == '__main__':
    start_app()
