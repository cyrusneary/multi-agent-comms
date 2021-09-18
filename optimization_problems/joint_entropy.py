import cvxpy as cp
from cvxpy.settings import NONNEG
import numpy as np

import sys
sys.path.append('../')
from markov_decision_process.mdp import MDP

def build_joint_entropy_program(mdp: MDP, 
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

    ##### Define the problem variables
    x = cp.Variable(shape=(mdp.Ns, mdp.Na),
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

    constraints.append(occupancy_out - occupancy_init - occupancy_in == 0)

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

    entropy1 = -cp.sum(cp.rel_entr(x[mdp.active_states, :], y))
    entropy2 = cp.sum(cp.hstack(
                [cp.sum(cp.multiply(x[mdp.active_states, :], 
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