import cvxpy as cp
import numpy as np

import sys
sys.path.append('../')
from markov_decision_process.mdp import MDP

def build_reachability_LP(mdp: MDP, exp_len_coef : float = 1.0):
    """
    Construct an LP to solve for a set of occupancy measures that
    respect the MDP dynamics while maximizing the probability of
    reaching the target set.

    Parameters
    ----------
    mdp :
        An object representing the MDP on which the reachability problem
        is to be solved.

    Returns
    -------
    prob : cvxpy Problem
        The linear program to be solved.
    x : cvxpy Variable
        A Ns x Na variable where x[s,a] represents the occupancy
        measure of state-action pair (s,a).
    """

    ##### Define the problem variables
    x = cp.Variable(shape=(mdp.Ns, mdp.Na), name='x')
    exp_len_coef = cp.Parameter(name='expLenCoef', value=exp_len_coef)

    ##### Define the problem constraints
    constraints = []

    # All occupancy measures are positive
    constraints.append(x[mdp.active_states, :] >= 0)

    # Absorbing states are set to have no occupancy measure
    constraints.append(x[mdp.absorbing_states, :] == 0)

    # The remaining states satisfy the balance equations
    occupancy_out = cp.sum(x[mdp.active_states, :], axis=1)

    occupancy_init = np.zeros((mdp.Ns,))
    occupancy_init[mdp.initial_state] = 1.0
    occupancy_init = occupancy_init[mdp.active_states]

    occupancy_in = cp.hstack(
        [cp.sum(cp.multiply(x[mdp.active_states, :], 
                            mdp.T[mdp.active_states, :, i])) 
            for i in mdp.active_states]
        )

    constraints.append(occupancy_out - occupancy_init - occupancy_in == 0)

    ##### Define the problem objective
    target_set_in = cp.sum(cp.hstack(
            [cp.sum(cp.multiply(x[mdp.active_states, :], 
                                mdp.T[mdp.active_states, :, i])) 
                for i in mdp.target_set]
            )
        )

    expected_length = cp.sum(x)

    obj = cp.Maximize(target_set_in - exp_len_coef * expected_length)

    ##### Create the problem
    prob = cp.Problem(obj, constraints)

    return prob, x

def process_occupancy_vars(x : cp.Variable):
    """
    Make sure all of the occupancy variables are positive.
    It's sometimes possible for the occupancy variables to be very small 
    negative numbers due to numerical errors.
    """
    x = x.value
    Ns, Na = x.shape
    for s in range(Ns):
        for a in range(Na):
            if x[s,a] < 0.0:
                assert np.abs(x[s,a]) <= 1e-10
                x[s,a] = 0.0
    return x