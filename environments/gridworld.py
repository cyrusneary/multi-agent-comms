import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
sys.path.append('../')

from markov_decision_process.mdp import MDP
from optimization_problems.reachability_LP import build_reachability_LP, \
                                                    process_occupancy_vars

class Gridworld(object):

    def __init__(self, 
                Nr : int = 5, 
                Nc : int = 5,
                slip_p : float = 0.05,
                initial_state : tuple = (0,0),
                target_states : list = [(4,4)],
                dead_states : list = [],
                walls : list = []
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
            Position of the initial state of the gridworld.
        target_states :
            List of positions of target states in the gridworld.
        dead_states :
            List of positions of dead states in the gridworld.
        walls :
            List of positions of the walls in the gridworld.
        """

        self.Nr = Nr
        self.Nc = Nc
        self.Ns = self.Nr * self.Nc

        self._construct_state_space()

        self.initial_state = initial_state
        self.initial_index = self.index_from_pos[initial_state]
        assert self.initial_index <= self.Ns - 1

        assert len(target_states) > 0
        self.target_states = target_states
        self.target_indexes = [self.index_from_pos[t_state] 
                                for t_state in self.target_states]

        assert slip_p <= 1.0
        self.slip_p = slip_p

        self.dead_states = dead_states
        self.dead_indexes = [self.index_from_pos[d_state]
                                for d_state in self.dead_states]

        self.walls = walls
        self.wall_indexes = [self.index_from_pos[w_state]
                                for w_state in self.walls]

        # Actions:
        # 0 : right
        # 1 : up
        # 2 : left
        # 3 : down
        self.Na = 4

        self._build_transition_matrix()

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
        gridworld environment. 
        
        self.T[s,a,s'] = probability of reaching state s' from state s
                            under action a.
        
        Given an action, the environment proceeds to the desired next 
        state with prob 1.0 - slip_p. The remaining slip_p is 
        distributed among the remaining adjacent states. If the selected
        action moves into a wall, then all of the probability is 
        assigned to the available adjacent states.
        """
        self.T = np.zeros((self.Ns, self.Na, self.Ns))

        for s in range(self.Ns):

            # Check if the state is absorbing before assigning 
            # any probability values.
            if (s in self.target_indexes) or (s in self.dead_states):
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
        for state in self.target_indexes:
            for action in range(self.Na):
                self.T[state, action, state] = 1.0

        # Set all dead states to be absorbing
        for state in self.dead_states:
            for action in range(self.Na):
                self.T[state, action, state] = 1.0

    def build_mdp(self, gamma : float = 1.0):
        """Build an MDP model of the environment."""
        return MDP(self.Nr * self.Nc, 
                    self.Na, 
                    self.T, 
                    self.initial_index,
                    self.target_indexes,
                    self.dead_states,
                    gamma=gamma
                    )

    def display(self, state=None, ax=None, plot=False, highlighted_states=None):

        # Display parameters
        grid_spacing = 1
        max_x = grid_spacing * self.Nc
        max_y = grid_spacing * self.Nr

        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, aspect='equal')
            plot = True

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Plot gridworld lines
        for i in range(self.Nr + 1):
            ax.plot([0, max_x], [- i * grid_spacing, - i * grid_spacing], color='black')
        for j in range(self.Nc + 1):
            ax.plot([j * grid_spacing, j * grid_spacing], [0, -max_y], color='black')

        # Plot the initial state
        (init_r, init_c) = self.initial_state
        ax.plot(init_c * grid_spacing + grid_spacing/2, 
                - (init_r * grid_spacing + grid_spacing/2), 
                linestyle=None, marker='x', markersize=15, color='blue')

        # plot the current state
        if state is not None:
            (state_r, state_c) = state
            ax.plot(state_c * grid_spacing + grid_spacing/2,
                    -(state_r * grid_spacing + grid_spacing/2), 
                    linestyle=None, marker='o', markersize=20, color='blue')

        # Plot the target locations
        for goal in self.target_states:
            (goal_r, goal_c) = goal
            goal_square = patches.Rectangle((goal_c, 
                                            -(goal_r + 1) * grid_spacing), 
                                            grid_spacing, grid_spacing, 
                                            fill=True, color='green')
            ax.add_patch(goal_square)

        # Plot walls
        for wall in self.walls:
            (wall_r, wall_c) = wall
            wall_square = patches.Rectangle((wall_c * grid_spacing, 
                                            -(wall_r + 1) * grid_spacing), 
                                            grid_spacing, grid_spacing, 
                                            fill=True, color='black')
            ax.add_patch(wall_square)

        # Plot the obstacles
        for obstacle in self.dead_states:
            (obs_r, obs_c) = obstacle
            obs_square = patches.Rectangle((obs_c * grid_spacing, 
                                            -(obs_r + 1) * grid_spacing), 
                                            grid_spacing, grid_spacing, 
                                            fill=True, color='red')
            ax.add_patch(obs_square)

        if plot:
            plt.show()

    def visualize_occupancy_measures(self, 
                                    occupancy_vals, 
                                    ax=None, 
                                    plot=True, 
                                    color='black', 
                                    highlighted_states=None):

        # Display parameters
        grid_spacing = 1
        max_x = grid_spacing * self.Nc
        max_y = grid_spacing * self.Nr

        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, aspect='equal')
            plot = True

        self.display(ax=ax, highlighted_states=highlighted_states)

        max_occupancy_val = np.max(occupancy_vals)

        for s in range(self.Ns):
            if not(s in self.wall_indexes) \
                and not(s in self.dead_indexes) \
                    and not(s in self.target_indexes):

                row, col = self.pos_from_index[s]

                for action in range(self.Na):
                    x = occupancy_vals[s, action]
                    if action == 0: # trying to move right
                        next_row = row
                        next_col = col + 1

                    if action == 1: # trying to move up
                        next_row = row - 1
                        next_col = col

                    if action == 2: # trying to move left
                        next_row = row
                        next_col = col - 1

                    if action == 3: # trying to move down
                        next_row = row + 1
                        next_col = col

                    start_x = col * grid_spacing + grid_spacing/2
                    start_y = -(row * grid_spacing + grid_spacing/2)

                    dx = (next_col - col) * grid_spacing
                    dy = -(next_row - row) * grid_spacing

                    ax.arrow(start_x, start_y, dx, dy,
                            shape='full', linewidth=2.0, 
                            length_includes_head=True,
                            head_width=0.2, color=color, 
                            alpha=np.log(x/max_occupancy_val+1.0))

        if plot:
            plt.show()

def main():
    # Build the gridworld
    gridworld = Gridworld(Nr=5, Nc=5)

    # Construct the corresponding MDP
    mdp = gridworld.build_mdp()

    # Construct and solve the reachability LP
    prob, x = build_reachability_LP(mdp)
    prob.parameters()[0].value = 0.1
    prob.solve()
    print(prob.solution.opt_val)

    # Visualize the occupancy measures
    occupancy_vars = process_occupancy_vars(x)
    gridworld.visualize_occupancy_measures(occupancy_vars)

    # Solve for the corresponding policy
    policy = mdp.policy_from_occupancy_vars(occupancy_vars)
    print(policy)

if __name__ == '__main__':
    main()