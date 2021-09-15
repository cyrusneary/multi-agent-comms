import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from mdp import MDP

class Gridworld(object):

    def __init__(self, 
                Nr : int = 5, 
                Nc : int = 5,
                slip_p : float = 0.05,
                initial_state : int = 0,
                target_states : list = [24],
                dead_states : list = []
                ):
        """
        Initializer for the Gridworld object.

        Parameters
        ----------
        Nr :
            Number of gridworld rows.
        Nc :
            Number of gridworld columns.
        slip_p :
            Slip probability. Should be 0 <= slip_p <= 1
        initial_state :
            Index of the initial state of the gridworld.
        target_states :
            List of indexes of target states in the gridworld.
        dead_states :
            List of indexes of dead states in the gridworld.
        """

        self.Nr = Nr
        self.Nc = Nc
        self.Ns = self.Nr * self.Nc

        self._construct_state_space()

        assert initial_state <= self.Ns - 1
        self.initial_state = initial_state

        assert len(target_states) > 0
        self.target_states = target_states

        assert slip_p <= 1.0
        self.slip_p = slip_p

        self.dead_states = dead_states

        # Actions:
        # 0 : right
        # 1 : up
        # 2 : left
        # 3 : down
        self.Na = 4

        self._build_transition_matrix()
        self._build_mdp()

    def _construct_state_space(self):
        """
        Build two maps providing state indexes from (row,col) positions
        and vice versa.
        """
        self.index_from_pos = {}
        self.pos_from_index = {}

        for i in range(self.Ns):
            self.pos_from_index[i] = np.unravel_index(i, (self.Nr, self.Nc))
            self.index_from_pos[self.pos_from_index[i]] = i

    def _build_transition_matrix(self):
        """
        Build the transition matrix storing the dynamics of the 
        gridworld environment. Given an action, the environment 
        proceeds to the desired next state with prob 1.0 - slip_p.
        The remaining slip_p is distributed among the remaining adjacent
        states. If the selected action moves into a wall, then all of 
        the probability is assigned to the available adjacent states.
        """
        self.T = np.zeros((self.Ns, self.Na, self.Ns))

        for s in range(self.Ns):

            # Check if the state is absorbing before assigning 
            # any probability values.
            if (s in self.target_states) or (s in self.dead_states):
                continue

            state_r, state_c = self.pos_from_index[s]
            
            # Create a dictionary of the available actions and the
            # corresponding next (intended) state
            avail_next_states = {}
            if state_c + 1 < self.Nc:
                avail_next_states[0] = (state_r, state_c + 1) # right
            if state_c - 1 >= 0:
                avail_next_states[2] = (state_r, state_c - 1) # left
            if state_r - 1 >= 0:
                avail_next_states[1] = (state_r - 1, state_c) # up
            if state_r + 1 < self.Nr:
                avail_next_states[3] = (state_r + 1, state_c) # down

            # action 0: move right
            # action 1: move up
            # action 2: move left
            # action 3: move down

            for a in range(self.Na):
                # First check if the action leads to a valid next state
                if a in avail_next_states.keys():
                    # Probability the action was executed successfully
                    s_next = self.index_from_pos[avail_next_states[a]]
                    self.T[s, a, s_next] = 1.0 - self.slip_p

                    # How many other possible states could we slip to?
                    num_slips = len(avail_next_states) - 1

                    # Probability of slipping to another neighboring state
                    for key, val in avail_next_states.items():
                        if not(key == a):
                            s_next = self.index_from_pos[val]
                            self.T[s, a, s_next] = self.slip_p / num_slips
                
                # This is the case where the action takes us into a wall.
                else: 
                    # How many other possible states could we slip to?
                    num_slips = len(avail_next_states)

                    # Probability of slipping to another neighboring state
                    for key, val in avail_next_states.items():
                        s_next = self.index_from_pos[val]
                        self.T[s, a, s_next] = 1.0 / num_slips

        # Set all target states to be absorbing
        for state in self.target_states:
            for action in range(self.Na):
                self.T[state, action, state] = 1.0

        # Set all dead states to be absorbing
        for state in self.dead_states:
            for action in range(self.Na):
                self.T[state, action, state] = 1.0

    def _build_mdp(self):
        """Build an MDP model of the environment."""
        self.mdp = MDP(self.Nr * self.Nc, 
                        self.Na, 
                        self.T, 
                        self.initial_state,
                        self.target_states,
                        self.dead_states,
                        )

def main():
    gridworld = Gridworld(Nr=5, Nc=5)
    gridworld.mdp.build_model()
    print(gridworld)

if __name__ == '__main__':
    main()