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
    int_step = 0.01
    while current_order < order:
        current_order = calculate_local_sync_order(_phases, G)
        results = odeint(f, _phases, np.arange(t, t + step, int_step))
        _phases = results[-1] % (2 * np.pi)
        # calculate_phases(_phases, time_counter, step, int_step)
        t += step
    print(t)
    return _phases
