import cvxpy as cp
import numpy as np

import sys
sys.path.append('../')
from markov_decision_process.mdp import MDP

def build_random_policy_program(mdp: MDP, 
                                rand_init : np.ndarray,
                                N_agents : int,
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
    N_agents :
        The number of agents.
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

    # Add an extra dummy action for each agent, that will be taken with 
    # probability 1 when the agent reaches any absorbing state.

    Na_local = round(pow(mdp.Na, 1/N_agents))
    Na = (Na_local + 1) ** N_agents

    action_shape = tuple([Na_local + 1 for i in range(N_agents)])
    
    local_auxiliary_action_index = Na_local

    true_actions = []
    partially_auxiliary_actions = []
    all_auxiliary_actions = []
    for a_joint in range(Na):
        a_tuple = np.unravel_index(a_joint, action_shape)
        if np.sum(np.array(a_tuple) == local_auxiliary_action_index) == 0:
            true_actions.append(a_joint)
        elif np.sum(np.array(a_tuple) == local_auxiliary_action_index) == N_agents:
            all_auxiliary_actions.append(a_joint)
        else:
            partially_auxiliary_actions.append(a_joint)

    ##### Define the problem variables
    x = cp.Variable(shape=(mdp.Ns, Na), name='x')

    vars = [x]
    params = []

    ##### Define the problem constraints
    constraints = []

    # All occupancy measures are positive, except for the dummy action.
    ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(true_actions)))
    ind_cols = np.tile(np.array(true_actions), (len(mdp.active_states), 1))
    constraints.append(x[ind_rows, ind_cols] >= 0)

    ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(partially_auxiliary_actions)))
    ind_cols = np.tile(np.array(partially_auxiliary_actions), (len(mdp.active_states), 1))
    constraints.append(x[ind_rows, ind_cols] == 0)

    ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(all_auxiliary_actions)))
    ind_cols = np.tile(np.array(all_auxiliary_actions), (len(mdp.active_states), 1))
    constraints.append(x[ind_rows, ind_cols] == 0)

    # From the absorbing states, only the dummy action has non-zero measure.
    ind_rows = np.tile(np.array([mdp.absorbing_states]).transpose(), (1, len(true_actions)))
    ind_cols = np.tile(np.array(true_actions), (len(mdp.absorbing_states), 1))
    constraints.append(x[ind_rows, ind_cols] == 0)

    ind_rows = np.tile(np.array([mdp.absorbing_states]).transpose(), (1, len(partially_auxiliary_actions)))
    ind_cols = np.tile(np.array(partially_auxiliary_actions), (len(mdp.absorbing_states), 1))
    constraints.append(x[ind_rows, ind_cols] == 0)

    ind_rows = np.tile(np.array([mdp.absorbing_states]).transpose(), (1, len(all_auxiliary_actions)))
    ind_cols = np.tile(np.array(all_auxiliary_actions), (len(mdp.absorbing_states), 1))
    constraints.append(x[ind_rows, ind_cols] >= 0)

    # The remaining states satisfy the balance equations
    occupancy_out = cp.sum(x[:, :], axis=1)

    occupancy_init = np.zeros((mdp.Ns,))
    occupancy_init[mdp.initial_state] = 1.0
    occupancy_init = occupancy_init[:]

    ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(true_actions)))
    ind_cols = np.tile(np.array(true_actions), (len(mdp.active_states), 1))
    occupancy_in = cp.hstack(
            [cp.sum(cp.multiply(x[ind_rows, ind_cols], 
                                mdp.T[mdp.active_states, :, i])) 
                for i in range(mdp.Ns)]
            )

    constraints.append(occupancy_out - occupancy_init - occupancy_in == 0)

    constraints.append(cp.sum(x) <= max_length_constr)

    ##### Define the problem objective
    obj = cp.Minimize(cp.sum(cp.power(x - rand_init, 2)))

    ##### Create the problem
    prob = cp.Problem(obj, constraints)

    return prob, vars, params