import numpy as np
import cvxpy as cp
from numpy.lib.function_base import average

class MDP(object):
    """Class representing an MDP"""

    def __init__(self, 
                num_states: int, 
                num_actions: int,
                transition: np.ndarray,
                initial_state: int,
                target_set: list,
                dead_states: list,
                gamma: float = 1.0,
                ):
        """
        
        Parameters
        ----------
        num_states : 
            Number of states of the MDP.
        num_actions :
            Total number of actions in the MDP.
        initial_state :
            Index of the initial state of the MDP.
        transition :
            Stochastic transition matrix of the MDP.
            transition[s,a,s'] is the probability of transitioning from
            state s to state s' under action a.
        target_set :
            Array of state indexes representing the target states.
        dead_states :
            Array of state indexes representing the absorbing states.
        gamma :
            Discount factor of the MDP.
        """
        assert num_states > 0 and num_actions > 0
        self.Ns = num_states
        self.Na = num_actions
        
        assert initial_state <= self.Ns - 1
        self.initial_state = initial_state

        assert transition.shape == (num_states, num_actions, num_states)
        self.T = transition

        assert np.max(target_set) <= self.Ns - 1
        self.target_set = target_set

        if len(dead_states) > 0: assert np.max(dead_states) <= self.Ns - 1
        self.dead_states = dead_states

        assert 0.0 <= gamma and gamma <= 1.0
        self.gamma = gamma

        # Get indexes of all absorbing and "active" states
        absorbing_states = set(self.target_set).union(set(self.dead_states))
        active_states = set(range(self.Ns)).difference(absorbing_states)
        self.absorbing_states = list(absorbing_states)
        self.active_states = list(active_states)

    def build_reachability_LP(self):
        """
        Construct an LP to solve for a set of occupancy measures that
        respect the MDP dynamics while maximizing the probability of
        reaching the target set.

        Returns
        -------
        prob : cvxpy Problem
            The linear program to be solved.
        x : cvxpy Variable
            A Ns x Na variable where x[s,a] represents the occupancy
            measure of state-action pair (s,a).
        """

        ##### Define the problem variables
        x = cp.Variable(shape=(self.Ns, self.Na), name='x')
        gamma = cp.Parameter(name='gamma', value=self.gamma)

        ##### Define the problem constraints
        constraints = []

        # All occupancy measures are positive
        constraints.append(x[self.active_states, :] >= 0)

        # Absorbing states are set to have no occupancy measure
        constraints.append(x[self.absorbing_states, :] == 0)

        # The remaining states satisfy the balance equations
        occupancy_out = cp.sum(x[self.active_states, :], axis=1)

        occupancy_init = np.zeros((self.Ns,))
        occupancy_init[self.initial_state] = 1.0
        occupancy_init = occupancy_init[self.active_states]

        occupancy_in = cp.hstack(
            [gamma * cp.sum(cp.multiply(x[self.active_states, :], 
                                self.T[self.active_states, :, i])) 
                for i in self.active_states]
            )

        constraints.append(occupancy_out - occupancy_init - occupancy_in == 0)

        ##### Define the problem objective
        target_set_in = cp.sum(cp.hstack(
                [cp.sum(cp.multiply(x[self.active_states, :], 
                                    self.T[self.active_states, :, i])) 
                    for i in self.target_set]
                )
            )

        obj = cp.Maximize(target_set_in)

        ##### Create the problem
        prob = cp.Problem(obj, constraints)

        return prob, x

    def policy_from_occupancy_vars(self, x : np.ndarray):
        """
        Build a policy from the occupancy measure values.

        Parameters
        ----------
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
        policy = np.zeros((self.Ns, self.Na))
        for s in self.active_states:
            for a in range(self.Na):
                policy[s,a] = x[s, a] / np.sum(x[s, :])
        return policy