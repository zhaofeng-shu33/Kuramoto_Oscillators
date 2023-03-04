import numpy as np
from scipy.integrate import odeint
import networkx as nx


def system_generator(G, nf):
    """ Generate the system of ODEs for the 
		Kuramoto's model of coupled oscillators 
	"""

    def f(theta, t=0):
        """ dtheta_i/dt = d(theta_i) 
		"""
        state = np.zeros(nf)
        for i in range(nf):
            for j in G[i]:
                state[i] += np.sin(theta[j] - theta[i])
            state[i] /= nf # * np.sin(Phi(t) - theta[i])
        return state

    return f



def calculate_local_sync_order(oscillator_phases, oscillatory_network):
    """!
    @brief Calculates level of local synchorization (local order parameter) for input phases for the specified network.
    @details This parameter is tend 1.0 when the oscillatory network close to local synchronization and it tend to 0.0 when 
                desynchronization is observed in the network.
    
    @param[in] oscillator_phases (list): List of oscillator phases that are used for level of local (partial) synchronization.
    @param[in] oscillatory_network (sync): Instance of oscillatory network whose connections are required for calculation.
    
    @return (double) Level of local synchronization (local order parameter).
    
    """

    exp_amount = 0.0
    for i, j in oscillatory_network.edges:
        exp_amount += np.exp(-abs(oscillator_phases[j] - oscillator_phases[i]))
    return exp_amount / oscillatory_network.number_of_edges()

def get_phase_value(G, order=0.99):
    """ Solves the system of ODEs describing the Kuramoto oscillators, 
		to determine the phases of the oscillators.
	"""
    nf = G.number_of_nodes()
    f = system_generator(G, nf)
    # initialize_phase
    _phases = np.random.uniform(0, 2 * np.pi, nf) % (
        2 * np.pi
    )  # restricting the angles to the unit circle
    current_order = 0.0
    t = 0
    step = 0.1
    while current_order < order:
        current_order = calculate_local_sync_order(_phases, G)
        results = odeint(f, _phases, np.arange(t, t + step, step))
        _phases = results[-1] % (2 * np.pi)
        # calculate_phases(_phases, time_counter, step, int_step)
        t += step
    print(t)
    return _phases

def allocate_sync_ensembles(last_state, tolerance=0.01):
    """!
    @brief Allocate clusters in line with ensembles of synchronous oscillators where each synchronous ensemble corresponds to only one cluster.
            
    @param[in] tolerance (double): Maximum error for allocation of synchronous ensemble oscillators.
    @param[in] indexes (list): List of real object indexes and it should be equal to amount of oscillators (in case of 'None' - indexes are in range [0; amount_oscillators]).
    @param[in] iteration (uint): Iteration of simulation that should be used for allocation.
    
    @return (list) Groups (lists) of indexes of synchronous oscillators.
            For example [ [index_osc1, index_osc3], [index_osc2], [index_osc4, index_osc5] ].
    
    """
    number_oscillators = len(last_state)
    clusters = [[0]]

    for i in range(1, number_oscillators, 1):
        cluster_allocated = False
        for cluster in clusters:
            for neuron_index in cluster:
                last_state_shifted = abs(last_state[i] - 2 * np.pi)

                if ( ( (last_state[i] < (last_state[neuron_index] + tolerance)) and (last_state[i] > (last_state[neuron_index] - tolerance)) ) or
                        ( (last_state_shifted < (last_state[neuron_index] + tolerance)) and (last_state_shifted > (last_state[neuron_index] - tolerance)) ) ):
                    cluster_allocated = True

                    real_index = i
                    cluster.append(real_index)
                    break

            if cluster_allocated is True:
                break

        if cluster_allocated is False:
            clusters.append([i])
    labels = [0] * number_oscillators
    for index, cluster in enumerate(clusters):
        for i in cluster:
            labels[i] = index
    return labels

def kuramoto_detection(G, k=None, method='kmeans'):
    phases = get_phase_value(G)
    if method == 'kmeans' and k is not None:
        from sklearn import cluster
        phases_ = ((phases - np.mean(phases)) / np.std(phases))
        labels = list(cluster.k_means(phases_.reshape(-1, 1), n_clusters=k)[1])
    else:
        tolerance = np.std(phases) / 2
        labels = allocate_sync_ensembles(phases, tolerance)
    return labels