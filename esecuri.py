def greedy_community_detectionf(adj_matrix):
    """
    Compute the community structure of a graph using a greedy algorithm.

    Args:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.

    Returns:
        np.ndarray: Array of community assignments for each node.
    """
    # Initialize each node in its own community
    n_nodes = adj_matrix.shape[0]
    communities = np.arange(n_nodes)

    # Compute the total number of edges in the graph
    m = np.sum(adj_matrix) / 2

    while True:
        # Compute the modularity matrix
        mod_matrix = compute_modularity_matrix(adj_matrix, communities, m)

        # Find the node that can be moved to another community to maximize modularity
        node_to_move, new_community = find_node_to_move(mod_matrix, communities)

        # If no node can be moved to increase modularity, terminate
        if node_to_move is None:
            break

        # Move the node to the new community
        communities[node_to_move] = new_community

    return communities
def compute_modularity_matrix(adj_matrix, communities, m):
    """
    Compute the modularity matrix of a graph given its adjacency matrix and community structure.

    Args:
        adj_matrix (np.ndarray): Adjacency matrix of the graph.
        communities (np.ndarray): Array of community assignments for each node.
        m (float): Total number of edges in the graph.

    Returns:
        np.ndarray: Modularity matrix.
    """
    k = np.sum(adj_matrix, axis=1)
    mod_matrix = adj_matrix - np.outer(k, k) / (2 * m) # Bij = Aij - (ki*kj) / 2*m
    mod_matrix *= np.equal.outer(communities, communities) # se inmulteste cu o matrice in care eij e 1 daca i si j sunt din aceeasi comunitate
    return mod_matrix
def find_node_to_move(mod_matrix, communities):
    """
    Find the node that can be moved to another community to maximize modularity.

    Args:
        mod_matrix (np.ndarray): Modularity matrix of the graph.
        communities (np.ndarray): Array of community assignments for each node.

    Returns:
        Tuple[int, int]: Index of the node to move and index of the new community.
        If no node can be moved to increase modularity, return (None, None).
    """
    max_delta_modularity = -np.inf
    node_to_move = None
    new_community = None

    for i in range(len(communities)):
        # Compute the increase in modularity that results from moving node i to each of its neighbors' communities
        delta_modularity = np.sum(mod_matrix[i, np.equal(communities, communities[i])]) - np.sum(mod_matrix[i])
        if delta_modularity > max_delta_modularity:
            max_delta_modularity = delta_modularity
            node_to_move = i
            new_community = np.argmax(np.bincount(communities[np.nonzero(mod_matrix[node_to_move])]))

    if max_delta_modularity > 0:
        return node_to_move, new_community
    else:
        return None, None


def greedy_community_detection(G):
    # Precompute the degrees and total number of edges
    degrees = dict(G.degree())
    m = G.number_of_edges()

    # Compute the modularity matrix
    A = nx.adjacency_matrix(G)
    B = A - lil_matrix(np.outer(list(degrees.values()), list(degrees.values())) / (2 * m))

    # Start with each node in its own community
    communities = [{n} for n in G.nodes()]
    node2com = {n: i for i, com in enumerate(communities) for n in com}

    # Repeat until no further improvement can be achieved
    improvement = True
    iteration = 0
    while improvement or iteration < 5:
        improvement = False
        iteration += 1
        for node in G.nodes():
            current_com = communities[node2com[node]]
            current_modularity = B[node, node] + 2 * B[node, :].sum() - 2 * B[node, list(current_com)].sum()
            best_com = node2com[node]
            best_delta = 0
            for neighbor in G.neighbors(node):
                neighbor_com = node2com[neighbor]
                if neighbor_com == best_com:
                    continue
                new_com = current_com.union(communities[neighbor_com])
                new_modularity = B[list(new_com), :][:, list(new_com)].sum()
                delta = new_modularity - current_modularity
                if delta > best_delta:
                    best_com = neighbor_com
                    best_delta = delta
            if best_delta > 0:
                # Move the node to its new community
                communities[node2com[node]] = current_com - {node}
                communities[best_com] = communities[best_com] | {node}
                node2com[node] = best_com
                improvement = True

    # Return the final set of communities
    return [list(c) for c in communities]


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


def modularity(adj_matrix, communities):
    # calculate the modularity of a partition
    m = adj_matrix.sum()
    k = adj_matrix.sum(axis=1)
    q = 0.0
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if communities[i] == communities[j]:
                q += adj_matrix[i,j] - k[i]*k[j]/(2*m)
    q /= (2*m)
    return q
def newman_greedy(adj_matrix):
    # find communities in a network using Newman's greedy algorithm
    n = adj_matrix.shape[0]
    communities = np.arange(n)
    q = modularity(adj_matrix, communities)
    while True:
        max_delta_q = 0
        max_i = None
        max_j = None
        for i in range(n):
            for j in range(i+1,n):
                if communities[i] != communities[j]:
                    new_communities = communities.copy()
                    new_communities[new_communities==communities[j]] = communities[i]
                    new_q = modularity(adj_matrix, new_communities)
                    delta_q = new_q - q
                    if delta_q > max_delta_q:
                        max_delta_q = delta_q
                        max_i = i
                        max_j = j
        if max_delta_q > 0:
            communities[communities == communities[max_j]] = communities[max_i]
            q += max_delta_q
        else:
            break
    return communities


def modularity(adj_matrix, communities, m):
    # calculate the modularity of a partition
    k = adj_matrix.sum(axis=1)
    q = 0.0
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if communities[i] == communities[j]:
                q += adj_matrix[i,j] - k[i]*k[j]/(2*m)
    q /= (2*m)
    return q
def louvain_community_detection(adj_matrix):
    # initialize each node to its own community
    n = adj_matrix.shape[0]
    communities = np.arange(n)
    # initialize the modularity
    m = adj_matrix.sum() / 2
    q = modularity(adj_matrix, communities, m)
    # loop until convergence
    converged = False
    while not converged:
        converged = True
        # loop over all nodes and their communities
        for i in range(n):
            current_community = communities[i]
            k_in = np.sum(adj_matrix[i, communities == current_community])
            k_i = np.sum(adj_matrix[i,:])
            q_i = (k_in - k_i/(2*m)) / (2*m)
            # loop over neighboring communities and their modularity gains
            community_counts = np.bincount(communities)
            community_sums = np.zeros(community_counts.shape)
            np.add.at(community_sums, communities, adj_matrix[i,:])
            community_sums -= adj_matrix[i, current_community]
            for j in np.unique(communities):
                if j != current_community:
                    k_j = community_sums[j]
                    k_in_j = np.sum(adj_matrix[i, communities == j])
                    q_j = (k_in_j - k_i*k_j/(2*m)) / (2*m)
                    delta_q = q_j - q_i
                    # if the modularity gain is positive, move the node to the new community
                    if delta_q > 0:
                        communities[i] = j
                        q += delta_q
                        converged = False
        # if the community assignment has not changed, break the loop
        if converged:
            break
        # merge nodes in the same community
        unique_communities = np.unique(communities)
        new_communities = np.zeros_like(communities)
        for i, comm in enumerate(unique_communities):
            nodes_in_comm = np.where(communities == comm)[0]
            new_communities[nodes_in_comm] = i
        communities = new_communities
        m = adj_matrix.sum() / 2
    return communities