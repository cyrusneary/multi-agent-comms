from ntpath import join
import cvxpy as cp
from cvxpy.lin_ops.lin_utils import conv
from cvxpy.settings import NONNEG
import numpy as np
from scipy.special import rel_entr, entr

import sys, os, time
sys.path.append('../')
from markov_decision_process.mdp import MDP

def build_linearized_program(mdp: MDP, 
                            N_agents : int,
                            agent_state_size_list : list,
                            agent_action_size_list : list,
                            check_agent_state_action_with_aux,
                            check_agent_state,
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
    check_agent_state_action_with_aux :
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

    # Add a dummy action

    # Define index matrices that will be very helpful for computing
    # the gradients of the entropy terms of the mixed processes for each
    # agent.

    f_grad = {}
    g_grad = {}

    # Add an extra dummy action for each agent, that will be taken with 
    # probability 1 when the agent reaches any absorbing state.
    agent_action_size_list_aux = []
    for i in range(N_agents):
        agent_action_size_list_aux.append(agent_action_size_list[i] + 1)
    Na = 1
    for i in range(N_agents):
        Na = Na * agent_action_size_list_aux[i]

    action_shape = tuple([agent_action_size_list_aux[i] for i in range(N_agents)])
    
    local_auxiliary_action_index = agent_action_size_list_aux[0] - 1

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

    # Build the gradient vectors.
    print('Building gradient vectors.')
    t_start = time.time()
    for i in range(N_agents):
        f_grad[i] = {}
        g_grad[i] = {}
        for s in range(agent_state_size_list[i]):

            g_grad[i][s] = np.zeros((mdp.Ns, Na))
            for s_joint in range(mdp.Ns):
                for a_joint in range(Na):
                    if check_agent_state(i, s, s_joint):
                        g_grad[i][s][s_joint, a_joint] = 1.0

            f_grad[i][s] = {}
            for a in range(agent_action_size_list_aux[i]):
                f_grad[i][s][a] = np.zeros((mdp.Ns, Na))
                for s_joint in range(mdp.Ns):
                    for a_joint in range(Na):
                        if check_agent_state_action_with_aux(i, 
                                                    s, s_joint, 
                                                    a, a_joint):
                            f_grad[i][s][a][s_joint, a_joint] = 1.0
    print('Finished building gradients in {} seconds.'.format(time.time() 
                                                                - t_start))

    ##### Define the problem variables
    x = cp.Variable(shape=(mdp.Ns, Na),
                     name='x',
                     nonneg=True)
    tc = cp.Variable(shape=(1,),
                        name='tc', nonneg=False)
    g = cp.Variable(shape=(mdp.Ns, Na),
                    name='g',
                    nonneg=False)
    ct = cp.Variable(shape=(1,), name='ct', nonneg=False)
    reachability_coef = cp.Parameter(name='reachabilityCoef',
                                        value=reachability_coef,
                                        nonneg=True)
    exp_len_coef = cp.Parameter(name='expLenCoef', 
                                value=exp_len_coef,
                                nonneg=True)
    total_corr_coef = cp.Parameter(name='entropyCoef', 
                                value=total_corr_coef,
                                nonneg=True)
    x_last = cp.Parameter(shape=(mdp.Ns, Na),
                                name='x_last',
                                value=np.zeros((mdp.Ns, Na)),
                                nonneg=True)

    vars = [x, tc, g, ct]
    params = [reachability_coef, exp_len_coef, total_corr_coef, x_last]

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

    constraints.append(cp.sum(x[:, :]) <= max_length_constr)

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

    # occupancy_out = cp.sum(x[mdp.active_states, :], axis=1)

    # occupancy_init = np.zeros((mdp.Ns + 1,))
    # occupancy_init[mdp.initial_state] = 1.0
    # occupancy_init = occupancy_init[mdp.active_states]

    # occupancy_in = cp.hstack(
    #     [cp.sum(cp.multiply(x[mdp.active_states, :], 
    #                         mdp.T[mdp.active_states, :, i])) 
    #         for i in mdp.active_states]
    #     )

    constraints.append(occupancy_out 
                        - occupancy_init 
                        - occupancy_in == 0)

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

    y = cp.sum(x[:, :], axis=1)
    y = cp.vstack([y for i in range(Na)]).T

    joint_entropy = -cp.sum(cp.rel_entr(x[:, :], y))

    # y = cp.sum(x[mdp.active_states, :], axis=1)
    # y = cp.vstack([y for i in range(Na)]).T

    # joint_entropy = -cp.sum(cp.rel_entr(x[mdp.active_states, :], y))

    grad = 0.0
    convex_term = 0.0
    for i in range(N_agents):
        for s_local in range(agent_state_size_list[i]):
            for a_local in range(agent_action_size_list_aux[i]):
                numerator = cp.sum(x_last[np.where(f_grad[i][s_local][a_local])])
                denominator = cp.sum(x_last[np.where(g_grad[i][s_local])])
                ratio_term = numerator / denominator
                log_term = cp.log(ratio_term)

                grad = grad + log_term * f_grad[i][s_local][a_local]

                convex_term = convex_term + numerator * log_term

    constraints.append(ct == convex_term)
    constraints.append(g == grad)

    linearized_term = - convex_term - cp.sum(cp.multiply(grad, (x - x_last)))

    total_corr = linearized_term - joint_entropy
    constraints.append(tc >= total_corr)

    obj = cp.Maximize(reachability_coef * target_set_in 
                        - exp_len_coef * expected_length
                        - total_corr_coef * tc)

    ##### Create the problem
    prob = cp.Problem(obj, constraints)

    return prob, vars, params, f_grad, g_grad

############ OPTIMIZATION STATISTICS FUNTIONS

def process_occupancy_vars(x : cp.Variable):
    """
    Make sure all of the occupancy variables are at least small 
    positive values. It's sometimes possible for the occupancy variables 
    to be very small negative numbers due to numerical errors. Making 
    them not exactly zero also helps to avoid divide by zero errors
    in subsequent optimization steps.
    """
    x = x.value
    x[np.where(x <= 1e-10)] = 1e-10
    return x

def policy_from_occupancy_vars(mdp : MDP, x : np.ndarray, N_agents : int):
    """
    Build a policy from the occupancy measure values.

    Parameters
    ----------
    mdp :
        An object representing the MDP on which the reachability problem
        is to be solved.
    x :
        Array built such that x[s,a] represents the occupancy 
        measure of state-action pair (s,a).
    N_agents :
        The number of agents.

    Returns
    -------
    policy : ndarray
        Array built such that policy[s,a] represents the probability
        of taking action a from state s under this policy.
        Note that policy[s,a] = 0 for all a if s is absorbing.
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

    x_mod = np.copy(x)[:, true_actions]

    policy = np.zeros((mdp.Ns, mdp.Na))
    for s in range(mdp.Ns):
        for a in range(mdp.Na):
            if not (np.sum(x_mod[s,:]) == 0.0):
                policy[s,a] = x_mod[s, a] / np.sum(x_mod[s, :])
            else:
                policy[s,a] = 1.0 / len(x_mod[s,:])
    return policy

def success_probability_from_occupancy_vars(mdp : MDP, x : np.ndarray, N_agents : int):
    """
    Compute the probability of reaching the target set from the 
    initial state.

    Parameters
    ----------
    mdp :
        An object representing the MDP on which the reachability problem
        is to be solved.
    x :
        Array built such that x[s,a] represents the occupancy 
        measure of state-action pair (s,a).
    N_agents :
        The number of agents.

    Returns
    -------
    success_prob : float
        Probability of reaching the target set from the initial state.
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

    ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(true_actions)))
    ind_cols = np.tile(np.array(true_actions), (len(mdp.active_states), 1))
    succes_prob = np.sum(np.hstack(
        [np.sum(np.multiply(x[ind_rows, ind_cols], 
                            mdp.T[mdp.active_states, :, i]))
                            for i in mdp.target_set]
        ))
    return succes_prob

def expected_len_from_occupancy_vars(mdp : MDP, x : np.ndarray):
    """
    Compute the expected length of the trajectories, before either
    reaching the target set or crashing. 

    Parameters
    ----------
    mdp :
        An object representing the MDP on which the reachability problem
        is to be solved.
    x :
        Array built such that x[s,a] represents the occupancy 
        measure of state-action pair (s,a).

    Returns
    -------
    expected_len : float
        Expected length of the trajectory.
    """
    return np.sum(x)

def compute_joint_entropy(mdp : MDP, x : np.ndarray, N_agents : int):
    """
    Compute the entropy of the trajectories of joint states.

    Parameters
    ----------
    mdp :
        An object representing the MDP on which the reachability problem
        is to be solved.
    x :
        Array built such that x[s,a] represents the occupancy 
        measure of state-action pair (s,a).
    N_agents :
        The number of agents.

    Returns
    -------
    joint_entropy : float
        Entropy value of the distribution of trajectories of 
        joint states.
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

    y = np.sum(x[mdp.active_states, :], axis=1)
    y = np.vstack([y for i in range(Na)]).T

    entropy1 = -np.sum(rel_entr(x[mdp.active_states, :], y))

    ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(true_actions)))
    ind_cols = np.tile(np.array(true_actions), (len(mdp.active_states), 1))
    entropy2 = np.sum(np.hstack(
                [np.sum(np.multiply(x[ind_rows, ind_cols], 
                            entr(mdp.T[mdp.active_states, :, i])))
                    for i in range(mdp.Ns)]
                )
            )

    # joint_entropy = entropy1 + entropy2
    joint_entropy = entropy1

    return joint_entropy

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
    f_grad :
        A dictionary such that f_grad[agent_id][s_local][a_local][s,a]
        returns true if the local state-action pair (s_local, a_local)
        of agent agent_id agree with the joint state-action pair (s,a).
    g_grad :
        A dictionary such that f_grad[agent_id][s_local][s]
        returns true if the local state s_local of agent agent_id agree 
        with the joint state s.
    x :
        A (Ns, Na) array containing the occupancy measures of all the 
        joint state-action pairs in the MDP. 

    Return
    ------
    total_corr :
        The total correlation given the occupancy measure variables.
    """
    Na_local = round(pow(mdp.Na, 1/N_agents))
    Na = (Na_local + 1) ** N_agents

    # Add an extra dummy action for each agent, that will be taken with 
    # probability 1 when the agent reaches any absorbing state.
    agent_action_size_list_aux = []
    for i in range(N_agents):
        agent_action_size_list_aux.append(agent_action_size_list[i] + 1)

    y = np.sum(x[mdp.active_states, :], axis=1)
    y = np.vstack([y for i in range(Na)]).T

    joint_entropy = -np.sum(rel_entr(x[mdp.active_states, :], y))

    convex_term = 0.0
    for i in range(N_agents):
        for s_local in range(agent_state_size_list[i]):
            for a_local in range(agent_action_size_list_aux[i]):
                numerator = np.sum(x[np.where(f_grad[i][s_local][a_local])])
                denominator = np.sum(x[np.where(g_grad[i][s_local])])
                ratio_term = numerator / denominator
                log_term = np.log(ratio_term)

                convex_term = convex_term + numerator * log_term

    total_corr = - convex_term - joint_entropy
    return total_corr

    grad = 0.0
    convex_term = 0.0
    for i in range(N_agents):
        for s_local in range(agent_state_size_list[i]):
            for a_local in range(agent_action_size_list_aux[i]):
                numerator = cp.sum(x_last[np.where(f_grad[i][s_local][a_local])])
                denominator = cp.sum(x_last[np.where(g_grad[i][s_local])])
                ratio_term = numerator / denominator
                log_term = cp.log(ratio_term)

                grad = grad + log_term * f_grad[i][s_local][a_local]

                convex_term = convex_term + numerator * log_term

    linearized_term = - convex_term - cp.sum(cp.multiply(grad, (x - x_last)))