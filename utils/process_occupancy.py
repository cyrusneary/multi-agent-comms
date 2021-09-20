import cvxpy as cp
import numpy as np
from markov_decision_process.mdp import MDP
from scipy.special import rel_entr, entr

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

def policy_from_occupancy_vars(mdp : MDP, x : np.ndarray):
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

    Returns
    -------
    policy : ndarray
        Array built such that policy[s,a] represents the probability
        of taking action a from state s under this policy.
        Note that policy[s,a] = 0 for all a if s is absorbing.
    """
    policy = np.zeros((mdp.Ns, mdp.Na))
    for s in range(mdp.Ns):
        for a in range(mdp.Na):
            if not (np.sum(x[s,:]) == 0.0):
                policy[s,a] = x[s, a] / np.sum(x[s, :])
            else:
                policy[s,a] = 1.0 / len(x[s,:])
    return policy

def success_probability_from_occupancy_vars(mdp : MDP, x : np.ndarray):
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

    Returns
    -------
    success_prob : float
        Probability of reaching the target set from the initial state.
    """
    succes_prob = np.sum(np.hstack(
        [np.sum(np.multiply(x[mdp.active_states, :], 
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

def compute_joint_entropy(mdp : MDP, x : np.ndarray):
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

    Returns
    -------
    joint_entropy : float
        Entropy value of the distribution of trajectories of 
        joint states.
    """
    y = np.sum(x[mdp.active_states, :], axis=1)
    y = np.vstack([y for i in range(mdp.Na)]).T

    entropy1 = -np.sum(rel_entr(x[mdp.active_states, :], y))
    entropy2 = np.sum(np.hstack(
                [np.sum(np.multiply(x[mdp.active_states, :], 
                            entr(mdp.T[mdp.active_states, :, i])))
                    for i in range(mdp.Ns)]
                )
            )

    joint_entropy = entropy1 + entropy2

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