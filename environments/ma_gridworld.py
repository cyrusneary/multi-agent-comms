import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import time
import sys

sys.path.append('../')
from markov_decision_process.mdp import MDP

class MAGridworld(object):

    def __init__(self, 
                N_agents : int = 2,
                Nr : int = 5, 
                Nc : int = 5,
                slip_p : float = 0.05,
                initial_state : tuple = (4, 0, 4, 3),
                target_states : list = [(4, 4, 4, 1)],
                dead_states : list = [],
                walls : list = []
                ):
        """
        Initializer for the Multiagent Gridworld object.

        The team's states and actions are represented as:
        (row1, col1, row2, col2, ..., rowN, colN)
        (action1, action2, ..., actionN)

        Parameters
        ----------
        N_agents :
            Number of agents in the gridworld.
        Nr :
            Number of gridworld rows.
        Nc :
            Number of gridworld columns.
        slip_p :
            Slip probability. Should be 0 <= slip_p <= 1
        initial_state :
            The initial joint position of the agents in the gridworld
        target_states :
            List of joint positions of target states for the various 
            agents in the gridworld.
        dead_states :
            List of positions of dead states in the gridworld.
        walls :
            List of positions of the walls in the gridworld.
        """

        self.Nr = Nr
        self.Nc = Nc
        self.Ns_local = self.Nr * self.Nc
        self.N_agents = N_agents
        self.Ns_joint = (self.Ns_local)**self.N_agents

        # Actions:
        # 0 : right
        # 1 : up
        # 2 : left
        # 3 : down
        self.Na_local = 4
        self.Na_joint = self.Na_local**self.N_agents

        self._construct_state_space()
        self._construct_action_space()

        assert len(initial_state) == self.N_agents * 2
        self.initial_state = initial_state
        self.initial_index = self.index_from_pos[initial_state]
        assert self.initial_index <= self.Ns_joint - 1

        assert len(target_states) > 0
        self.target_states = target_states
        self.target_indexes = [self.index_from_pos[t_state] 
                                for t_state in self.target_states]

        assert slip_p <= 1.0
        self.slip_p = slip_p

        self.dead_states = dead_states
        self._construct_collision_dead_states()
        self.dead_indexes = [self.index_from_pos[d_state]
                                for d_state in self.dead_states]

        self.walls = walls
        self.wall_indexes = [self.index_from_pos[w_state]
                                for w_state in self.walls]

        self._build_transition_matrix()

    def save(self, save_file_str : str):
        """
        Save the multiagent gridworld object.
        """
        with open(save_file_str, 'wb') as f:
            pickle.dump(self, f) 

    def _construct_state_space(self):
        """
        Build two maps providing state indexes from 
        (row1,col1, ..., rowN, colN) positions and vice versa.
        """
        self.index_from_pos = {}
        self.pos_from_index = {}

        position_shape = tuple(self.Nr if i % 2 == 0 
                                else self.Nc for i in range(self.N_agents*2))

        for i in range(self.Ns_joint):
            self.pos_from_index[i] = np.unravel_index(i, position_shape)
            self.index_from_pos[self.pos_from_index[i]] = i

    def _construct_action_space(self):
        """
        Build two maps providing action indexes from (a1, a2, ..., aN)
        action specifications and vice versa.
        """
        self.action_index_from_tuple = {}
        self.action_tuple_from_index = {}

        action_shape = tuple(self.Na_local for i in range(self.N_agents))

        for i in range(self.Na_joint):
            self.action_tuple_from_index[i] = np.unravel_index(i, action_shape)
            self.action_index_from_tuple[self.action_tuple_from_index[i]] = i

    def _construct_collision_dead_states(self):
        """
        Add all collision states to the list of dead states.
        """
        for s in range(self.Ns_joint):
            state_tuple = self.pos_from_index[s]
            for agent_id in range(self.N_agents):
                for agent_id2 in range(agent_id + 1, self.N_agents):
                    if ((state_tuple[2*agent_id:(2*agent_id+2)] 
                        == state_tuple[2*agent_id2:(2*agent_id2+2)])
                        and not (state_tuple in self.dead_states)):

                        self.dead_states.append(state_tuple)


    def _build_transition_matrix(self):
        """
        Build the transition matrix storing the dynamics of the 
        gridworld environment. 
        
        self.T[s,a,s'] = probability of reaching joint state s' from 
                            joint state s under joint action a.

        The Local transitions of each agent are assumed to be independent.
        
        Given an action, each agent proceeds to the desired next 
        state with prob 1.0 - slip_p. The remaining slip_p is 
        distributed among the remaining adjacent states. If the selected
        action moves into a wall, then all of the probability is 
        assigned to the available adjacent states.
        """
        self.T = np.zeros((self.Ns_joint, self.Na_joint, self.Ns_joint))

        for s in range(self.Ns_joint):

            # Check if the state is absorbing before assigning 
            # any probability values.
            if (s in self.target_indexes) or (s in self.dead_indexes):
                continue

            # Get the tuple of row and column positions of all agents
            pos = self.pos_from_index[s] 

            # For each agent, find the next available local positions
            # and the probabilities of the individual agents 
            # transitioning to those states.
            agent_next_states = {}
            local_trans_funcs = {}
            for agent_id in range(self.N_agents):
                # dictionary of possible next local states for current agent
                agent_next_states[agent_id] = {} 

                # dictionary containing local transition functions mapping
                # local_trans_funcs[agent_id][local_action][local_next_state] 
                #           = prob value
                local_trans_funcs[agent_id] = {}

                state_r, state_c = pos[2*agent_id:(2*agent_id + 2)]

                if state_c + 1 < self.Nc: # right
                    agent_next_states[agent_id][0] = (state_r, state_c + 1)
                if state_c - 1 >= 0: # left
                    agent_next_states[agent_id][2] = (state_r, state_c - 1)
                if state_r - 1 >= 0: # up
                    agent_next_states[agent_id][1] = (state_r - 1, state_c) 
                if state_r + 1 < self.Nr: # down
                    agent_next_states[agent_id][3] = (state_r + 1, state_c) 

                # action 0: move right
                # action 1: move up
                # action 2: move left
                # action 3: move down

                for a in range(self.Na_local):
                    local_trans_funcs[agent_id][a] = {}
                    # First check if the action leads to a valid next state
                    if a in agent_next_states[agent_id].keys():
                        intended_next_state = agent_next_states[agent_id][a]
                        # Probability the action was executed successfully
                        local_trans_funcs[agent_id][a][intended_next_state] \
                                 = 1.0 - self.slip_p

                        # How many other possible states could we slip to?
                        num_slips = len(agent_next_states[agent_id]) - 1

                        # Probability of slipping to another neighboring state
                        for key, val in agent_next_states[agent_id].items():
                            if not(key == a):
                                local_trans_funcs[agent_id][a][val] = \
                                    self.slip_p / num_slips

                    else:
                        # How many other possible states could we slip to?
                        num_slips = len(agent_next_states[agent_id])

                        # Probability of slipping to another neighboring state
                        for key, val in agent_next_states[agent_id].items():
                            local_trans_funcs[agent_id][a][val] \
                                = 1.0 / num_slips

            # Now that we have the local transition functions of all 
            # the agents, construct the joint transition function using
            # the assumption that all local transition probabilities
            # are independent.
            for a_ind in range(self.Na_joint):
                action_tuple = self.action_tuple_from_index[a_ind]

                for next_s_ind in range(self.Ns_joint):
                    next_s_tuple = self.pos_from_index[next_s_ind]

                    prob_transition = 1.0

                    for agent_id in range(self.N_agents): 
                        local_action = action_tuple[agent_id]
                        local_state = next_s_tuple[2*agent_id:(2*agent_id + 2)]

                        if (local_state in local_trans_funcs[agent_id]
                                                [local_action].keys()):
                            prob_local_trans = \
                                local_trans_funcs[agent_id][local_action][local_state]
                        else:
                            prob_transition = 0.0
                            break

                        prob_transition = prob_transition * prob_local_trans

                    self.T[s, a_ind, next_s_ind] = prob_transition
                    
        # Set all target states to be absorbing
        for state in self.target_indexes:
            for action in range(self.Na_joint):
                self.T[state, action, state] = 1.0

        # Set all dead states to be absorbing
        for state in self.dead_indexes:
            for action in range(self.Na_joint):
                self.T[state, action, state] = 1.0

    def build_mdp(self, gamma : float = 1.0):
        """Build an MDP model of the environment."""
        return MDP(self.Ns_joint, 
                    self.Na_joint, 
                    self.T, 
                    self.initial_index,
                    self.target_indexes,
                    self.dead_indexes,
                    gamma=gamma
                    )

    #################### Visualization Methods

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

        for s in range(self.Ns_joint):
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

    print('Building gridworld')
    t_start = time.time()
    # Build the gridworld
    gridworld = MAGridworld(Nr=5, Nc=5)

    print('Constructed gridworld in {} seconds'.format(time.time() - t_start))

    for s in range(gridworld.Ns_joint):
        for a in range(gridworld.Na_joint):
            assert(np.abs(np.sum(gridworld.T[s, a, :]) - 1.0) <= 1e-12)

    # Construct the corresponding MDP
    mdp = gridworld.build_mdp()

    # Construct and solve the reachability LP
    prob, x = mdp.build_reachability_LP()
    prob.parameters()[0].value = 0.1
    prob.solve()
    print(prob.solution.opt_val)

    # with open('data.pickle', 'rb') as f:
    #     # The protocol version used is detected automatically, so we do not
    #     # have to specify it.
    #     data = pickle.load(f)

    # # Visualize the occupancy measures
    # occupancy_vars = mdp.process_occupancy_vars(x)
    # gridworld.visualize_occupancy_measures(occupancy_vars)

    # # Solve for the corresponding policy
    # policy = mdp.policy_from_occupancy_vars(occupancy_vars)
    # print(policy)

if __name__ == '__main__':
    main()