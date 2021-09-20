from ntpath import join
import cvxpy as cp
from cvxpy.lin_ops.lin_utils import conv
from cvxpy.settings import NONNEG
import numpy as np
from scipy.special import rel_entr

import sys, os, time
sys.path.append('../')
from markov_decision_process.mdp import MDP

def build_linearized_program(mdp: MDP, 
                            N_agents : int,
                            agent_state_size_list : list,
                            agent_action_size_list : list,
                            check_agent_state_action : function,
                            check_agent_state : function,
                            reachability_coef : float = 1.0,
                            exp_len_coef : float = 0.1,
                            total_corr_coef : float = 0.1,
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
        The number of agents involved in the problem.
    agent_state_size_list :
        A list of integers specifying the number of local states for
        each agent.
    agent_action_size_list :
        A list of integers specifying the number of local actions for
        each agent.
    check_agent_state_action :
        A function indicating whether or not the state-action pair of
        a particular agent agrees with a given joint state-action pair.
    check_agent_state :
        A function indicating whether or not the state pair of
        a particular agent agrees with a given joint state pair.
    exp_len_coef :
        Coefficient on the expected length term in the definition of
        the optimization objective.
    total_corr_coef :
        Coefficient on the total correlation term in the definition of 
        the optimization objective.
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
        params[1] is the total correlation coefficient used to define
        the optimization objective.
        params[2] is the last guess at a solution.
    """

    # Define index matrices that will be very helpful for computing
    # the gradients of the entropy terms of the mixed processes for each
    # agent.

    f_grad = {}
    g_grad = {}

    print('Building gradient vectors.')
    t_start = time.time()
    for i in range(N_agents):
        f_grad[i] = {}
        g_grad[i] = {}
        for s in range(agent_state_size_list[i]):

            g_grad[i][s] = np.zeros((mdp.Ns, mdp.Na))
            for s_joint in range(mdp.Ns):
                for a_joint in range(mdp.Na):
                    if check_agent_state(i, s, s_joint):
                        g_grad[i][s][s_joint, a_joint] = 1.0

            f_grad[i][s] = {}
            for a in range(agent_action_size_list[i]):
                f_grad[i][s][a] = np.zeros((mdp.Ns, mdp.Na))
                for s_joint in range(mdp.Ns):
                    for a_joint in range(mdp.Na):
                        if check_agent_state_action(i, 
                                                    s, s_joint, 
                                                    a, a_joint):
                            f_grad[i][s][a][s_joint, a_joint] = 1.0
    print('Finished building gradients in {} seconds.'.format(time.time() 
                                                                - t_start))

    ##### Define the problem variables
    x = cp.Variable(shape=(mdp.Ns, mdp.Na),
                     name='x',
                     nonneg=True)
    reachability_coef = cp.Parameter(name='reachabilityCoef',
                                        value=reachability_coef,
                                        nonneg=True)
    exp_len_coef = cp.Parameter(name='expLenCoef', 
                                value=exp_len_coef,
                                nonneg=True)
    total_corr_coef = cp.Parameter(name='entropyCoef', 
                                value=total_corr_coef,
                                nonneg=True)
    x_last = cp.Parameter(shape=(mdp.Ns, mdp.Na),
                                name='x_last',
                                value=np.zeros((mdp.Ns, mdp.Na)),
                                nonneg=True)

    vars = [x]
    params = [reachability_coef, exp_len_coef, total_corr_coef, x_last]

    ##### Define the problem constraints
    constraints = []

    # All occupancy measures are positive
    constraints.append(x[mdp.active_states, :] >= 0.0)

    # Absorbing states are set to have no occupancy measure
    constraints.append(x[mdp.absorbing_states, :] == 0.0)

    constraints.append(cp.sum(x) <= max_length_constr)

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

    constraints.append(occupancy_out 
                        - occupancy_init 
                        - occupancy_in == 0)

    ##### Define the problem objective
    target_set_in = cp.sum(cp.hstack(
            [cp.sum(cp.multiply(x[mdp.active_states, :], 
                                mdp.T[mdp.active_states, :, i])) 
                for i in mdp.target_set]
            )
        )

    expected_length = cp.sum(x)

    y = cp.sum(x[mdp.active_states, :], axis=1)
    y = cp.vstack([y for i in range(mdp.Na)]).T

    joint_entropy = -cp.sum(cp.rel_entr(x[mdp.active_states, :], y))

    grad = 0.0
    convex_term = 0.0
    for i in range(N_agents):
        for s_local in range(agent_state_size_list[i]):
            for a_local in range(agent_action_size_list[i]):
                numerator = cp.sum(x_last[np.where(f_grad[i][s_local][a_local])])
                denominator = cp.sum(x_last[np.where(g_grad[i][s_local])])
                ratio_term = numerator / denominator
                log_term = cp.log(ratio_term)

                grad = grad + log_term * f_grad[i][s_local][a_local]

                convex_term = convex_term + numerator * log_term

    linearized_term = - convex_term - cp.sum(cp.multiply(grad, (x - x_last)))

    total_corr = linearized_term - joint_entropy

    obj = cp.Maximize(reachability_coef * target_set_in 
                        - exp_len_coef * expected_length
                        - total_corr_coef * total_corr)

    ##### Create the problem
    prob = cp.Problem(obj, constraints)

    return prob, vars, params, f_grad, g_grad

def compute_total_correlation(mdp : MDP,
                                N_agents : int,
                                agent_state_size_list : list,
                                agent_action_size_list : list,
                                f_grad : dict,
                                g_grad : dict,
                                x : np.ndarray):
    """
    Compute the total correlation given the occupancy measure variables.

    Parameters
    ----------
    mdp :
        An object representing the MDP on which the reachability problem
        is to be solved.
    N_agents :
        The number of agents involved in the problem.
    agent_state_size_list :
        A list of integers specifying the number of local states for
        each agent.
    agent_action_size_list :
        A list of integers specifying the number of local actions for
        each agent.
    check_agent_state_action :
        A function indicating whether or not the state-action pair of
        a particular agent agrees with a given joint state-action pair.
    check_agent_state :
        A function indicating whether or not the state pair of
        a particular agent agrees with a given joint state pair.
    exp_len_coef :
        Coefficient on the expected length term in the definition of
        the optimization objective.
    total_corr_coef :
        Coefficient on the total correlation term in the definition of the
        optimization objective.
    max_length_constr : 
        Hard constraint on the maximum expected length of the 
        trajectories. This constraint is necessary to ensure that the 
        optimization algorithm terminates.

    Return
    ------
    total_corr :
        The total correlation given the occupancy measure variables.
    """
    y = np.sum(x[mdp.active_states, :], axis=1)
    y = np.vstack([y for i in range(mdp.Na)]).T

    joint_entropy = -np.sum(rel_entr(x[mdp.active_states, :], y))

    convex_term = 0.0
    for i in range(N_agents):
        for s_local in range(agent_state_size_list[i]):
            for a_local in range(agent_action_size_list[i]):
                numerator = np.sum(x[np.where(f_grad[i][s_local][a_local])])
                denominator = np.sum(x[np.where(g_grad[i][s_local])])
                ratio_term = numerator / denominator
                log_term = np.log(ratio_term)

                convex_term = convex_term + numerator * log_term

    total_corr = - convex_term - joint_entropy
    return total_corr

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
                assert np.abs(x[s,a]) <= 1e-9
                x[s,a] = 0.0
    return x