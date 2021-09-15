import numpy as np
import cvxpy as cp

class MDP(object):
    """Class representing an MDP"""

    def __init__(self, 
                num_states: int, 
                num_actions: int,
                transition: np.ndarray,
                initial_state: int,
                target_set: list,
                dead_states: list,
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

    def build_opt_problem(self):

        # Define the problem variables
        x = cp.Variable(shape=(self.Ns, self.Na), name='x')

        # Define the problem constraints
        constraints = []
        constraints.append(x >= 0)
        constraints.append(x[self.target_set] == 0)
        if len(self.dead_states > 0):
            constraints.append(x[self.dead_states] == 0)

    def policy_from_occupancy_vars(self, occupancy_vars):
        pass