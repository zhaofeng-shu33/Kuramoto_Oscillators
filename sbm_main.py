# SBM community detection by Kuramoto oscillators
import numpy as np
import networkx as nx
from sys_dynamics_functions import get_phase_value,\
    allocate_sync_ensembles, kuramoto_detection
from sklearn import metrics, cluster

def sbm_graph(n, k, a, b):
    # shuffled version to generate sbm graph
    if n % k != 0:
        raise ValueError('n %k != 0')
    elif a <= b:
        raise ValueError('a <= b')
    sizes = [int(n/k) for _ in range(k)]
    _p = np.log(n) * a / n
    _q = np.log(n) * b / n
    if _p > 1 or _q > 1:
        raise ValueError('%f (probability) larger than 1' % _p)
    G = nx.Graph()
    labels = sorted([i % k for i in range(n)])
    # np.random.shuffle(labels)
    for i in range(n):
        G.add_node(i, block=labels[i])
    for i in range(n):
        for j in range(i+1, n):
            u = np.random.uniform()
            if labels[i] == labels[j] and u <= _p:
                G.add_edge(i, j)
            elif labels[i] != labels[j] and u <= _q:
                G.add_edge(i, j)
    return G

def get_ground_truth(graph):
    label_list = []
    for n in graph.nodes(data=True):
        label_list.append(n[1]['block'])
    return label_list

def compare(label_0, label_1):
    '''
    get acc using adjusted rand index
    '''
    return metrics.adjusted_rand_score(label_0, label_1)

def main():
    """ Main program.
	"""

    # load network of oscillators
    k = 2
    G = sbm_graph(100, k, 16, 4)
    labels = kuramoto_detection(G)
    true_labels = get_ground_truth(G)
    print(compare(true_labels, labels))
    labels_kmeans = kuramoto_detection(G, 2)
    print(compare(true_labels, labels_kmeans))

if __name__ == "__main__":
    main()