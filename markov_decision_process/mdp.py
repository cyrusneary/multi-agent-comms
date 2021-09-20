import numpy as np
import cvxpy as cp
from numpy.lib.function_base import average
from scipy.special import rel_entr, entr

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