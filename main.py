import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import networkx as nx
from sys_dynamics_functions import get_phase_value
# from visualization import *


def load_2cluster_network():
    """ Generate graph with 12 nodes and 26 edges, with 2 
		main clusters connected by two paths
	"""
    G = nx.Graph()
    for i in range(12):
        G.add_node(i)
    edges_list = [(0, 1), (0, 3), (1, 3), (1, 4), (1, 2), (2, 4), (2, 3), (4, 5), (5, 8), (3, 10), (6, 7), (6, 8), (7, 9), (7, 8), (6, 9), (8, 9), (9, 10), (10, 11), (9, 11), (5, 1), (5, 2), (5, 0), (8, 10), (8, 11), (3, 5), (10, 6)]
    G.add_edges_from(edges_list)
    return G


def main():
    """ Main program.
	"""

    # load network of oscillators
    G = load_2cluster_network()
    phases = get_phase_value(G)
    print(phases)


if __name__ == "__main__":
    main()
