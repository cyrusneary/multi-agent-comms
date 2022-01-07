import cvxpy as cp
from cvxpy.settings import NONNEG
import numpy as np

import sys
sys.path.append('../')
from markov_decision_process.mdp import MDP

def build_joint_entropy_program(mdp: MDP, 
                                N_agents : int,
                                exp_len_coef : float = 0.1,
                                entropy_coef : float = 0.1,
                                max_length_constr : int = 100):
    """
    Construct a convex program to solve for a set of occupancy measures 
    that respect the MDP dynamics while maximizing the probability of
    reaching the target set.

    Parameters
    ----------
    mdp :
        An object representing the MDP on which the reachability problem
        is to be solved.
    N_agents :
        The number of agents.
    exp_len_coef :
        Coefficient on the expected length term in the definition of
        the optimization objective.
    entropy_coef :
        Coefficient on the entropy term in the definition of the
        optimization objective.
    max_length_constr : 
        Hard constraint on the maximum expected length of the 
        trajectories. This constraint is necessary to ensure that the 
        optimization algorithm terminates.

    Returns
    -------
    prob : cvxpy Problem
        The linear program to be solved.
    vars : list of cvxpy Variables
        vars[0] is a (Ns,Na) variable x, where x[s,a] represents the 
        occupancy measure of state-action pair (s,a).
    params : list of cvxpy Parameters
        params[0] is the expected length coefficient used to define
        the optimization objective.
        params[1] is the entropy coefficient used to define the
        optimization objective.
    """
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
    x = cp.Variable(shape=(mdp.Ns, Na),
                     name='x',
                     nonneg=True)
    exp_len_coef = cp.Parameter(name='expLenCoef', 
                                value=exp_len_coef,
                                nonneg=True)
    entropy_coef = cp.Parameter(name='entropyCoef', 
                                value=entropy_coef,
                                nonneg=True)

    vars = [x]
    params = [exp_len_coef, entropy_coef]

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

    ##### Define the problem objective
    ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(true_actions)))
    ind_cols = np.tile(np.array(true_actions), (len(mdp.active_states), 1))
    target_set_in = cp.sum(cp.hstack(
            [cp.sum(cp.multiply(x[ind_rows, ind_cols], 
                                mdp.T[mdp.active_states, :, i])) 
                for i in mdp.target_set]
            )
        )

    expected_length = cp.sum(x)

    constraints.append(expected_length <= max_length_constr)

    y = cp.sum(x[mdp.active_states, :], axis=1)
    y = cp.vstack([y for i in range(Na)]).T

    entropy1 = -cp.sum(cp.rel_entr(x[mdp.active_states, :], y))

    ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(true_actions)))
    ind_cols = np.tile(np.array(true_actions), (len(mdp.active_states), 1))
    entropy2 = cp.sum(cp.hstack(
                [cp.sum(cp.multiply(x[ind_rows, ind_cols], 
                            cp.entr(mdp.T[mdp.active_states, :, i])))
                    for i in range(mdp.Ns)]
                )
            )

    joint_entropy = entropy1 + entropy2

    obj = cp.Maximize(target_set_in 
                        - exp_len_coef * expected_length
                        + entropy_coef * joint_entropy)

    ##### Create the problem
    prob = cp.Problem(obj, constraints)

    return prob, vars, params