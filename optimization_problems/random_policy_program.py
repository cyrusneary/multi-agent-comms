import cvxpy as cp
import numpy as np

import sys
sys.path.append('../')
from markov_decision_process.mdp import MDP

def build_random_policy_program(mdp: MDP, 
                                rand_init : np.ndarray,
                                max_length_constr : int = 50):
    """
    Construct an LP to solve for a set of occupancy measures that
    respect the MDP dynamics while maximizing the probability of
    reaching the target set.

    Parameters
    ----------
    mdp :
        An object representing the MDP on which the reachability problem
        is to be solved.
    rand_init : 
        A randomly sampled initial guess for the occupancy measures
    max_length_constr : 
        Hard constraint on the maximum expected length of the 
        trajectories.

    Returns
    -------
    prob : cvxpy Problem
        The linear program to be solved.
    vars : list of cvxpy Variables
        vars[0] is a (Ns,Na) variable x, where x[s,a] represents the 
        occupancy measure of state-action pair (s,a).
    """

    ##### Define the problem variables
    x = cp.Variable(shape=(mdp.Ns, mdp.Na), name='x')

    vars = [x]
    params = []

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

    constraints.append(cp.sum(x) <= max_length_constr)

    ##### Define the problem objective
    obj = cp.Minimize(cp.sum(cp.power(x - rand_init, 2)))

    ##### Create the problem
    prob = cp.Problem(obj, constraints)

    return prob, vars, params