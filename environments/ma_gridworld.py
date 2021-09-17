from cvxpy.expressions.cvxtypes import index
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import numpy as np
import pickle
import sys, os, time

sys.path.append('../')
from markov_decision_process.mdp import MDP
from optimization_problems.reachability_LP import build_reachability_LP,\
                                                    process_occupancy_vars

class MAGridworld(object):

    def __init__(self, 
                N_agents : int = 2,
                Nr : int = 5, 
                Nc : int = 5,
                slip_p : float = 0.1,
                initial_state : tuple = (4, 0, 4, 4),
                target_states : list = [(4, 4, 4, 0)],
                dead_states : list = [],
                lava : list = [(0, 4), (0, 0)],
                walls : list = [],
                load_file_str : str = ''
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
        lava :
            List of positions of lava in the gridworld.
        walls :
            List of positions of the walls in the gridworld.
        load_file_str :
            String representing the path of a data file to use to load a
            pre-build multiagent gridworld object. If load_file_str = '', 
            then the gridworld is instead constructed from scratch.
        """

        if load_file_str == '':

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
            # 4 : stay
            self.Na_local = 5
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
            self.lava = lava
            self._construct_collision_dead_states()
            self._construct_lava_dead_states()
            self.dead_indexes = [self.index_from_pos[d_state]
                                    for d_state in self.dead_states]

            self.walls = walls

            self._build_transition_matrix()

        else:
            self.load(load_file_str)

    def save(self, save_file_str : str):
        """
        Save the multiagent gridworld object.
        """
        save_dict = {}
        save_dict['Nr'] = self.Nr
        save_dict['Nc'] = self.Nc
        save_dict['N_agents'] = self.N_agents
        save_dict['Ns_local'] = self.Ns_local
        save_dict['Ns_joint'] = self.Ns_joint
        save_dict['Na_local'] = self.Na_local
        save_dict['Na_joint'] = self.Na_joint
        save_dict['initial_state'] = self.initial_state
        save_dict['initial_index'] = self.initial_index
        save_dict['target_states'] = self.target_states
        save_dict['target_indexes'] = self.target_indexes
        save_dict['slip_p'] = self.slip_p
        save_dict['dead_states'] = self.dead_states
        save_dict['dead_indexes'] = self.dead_indexes
        save_dict['lava'] = self.lava
        save_dict['walls'] = self.walls
        save_dict['T'] = self.T

        with open(save_file_str, 'wb') as f:
            pickle.dump(save_dict, f) 

    def load(self, load_file_str : str):
        """
        Load the multiagent gridworld data from a file.
        """
        with open(load_file_str, 'rb') as f:
            save_dict = pickle.load(f)

        self.Nr= save_dict['Nr']
        self.Nc = save_dict['Nc']
        self.N_agents = save_dict['N_agents']
        self.Ns_local = save_dict['Ns_local']
        self.Ns_joint = save_dict['Ns_joint']
        self.Na_local = save_dict['Na_local']
        self.Na_joint = save_dict['Na_joint']
        self.initial_state = save_dict['initial_state']
        self.initial_index = save_dict['initial_index']
        self.target_states = save_dict['target_states']
        self.target_indexes = save_dict['target_indexes']
        self.slip_p = save_dict['slip_p']
        self.dead_states = save_dict['dead_states']
        self.dead_indexes = save_dict['dead_indexes']
        self.lava = save_dict['lava']
        self.walls = save_dict['walls']
        self.T = save_dict['T']

        self._construct_state_space()
        self._construct_action_space()

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

    def _construct_lava_dead_states(self):
        """
        Add all joint states corresponding to an agent being in lava.
        """
        for s in range(self.Ns_joint):
            state_tuple = self.pos_from_index[s]
            for agent_id in range(self.N_agents):
                if (state_tuple[2*agent_id:(2*agent_id + 2)] in self.lava
                    and not(state_tuple in self.dead_states)):
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

                if (state_c + 1 < self.Nc 
                        and not((state_r, state_c+1) in self.walls)): # right
                    agent_next_states[agent_id][0] = (state_r, state_c + 1)
                if (state_c - 1 >= 0
                        and not((state_r, state_c - 1) in self.walls)): # left
                    agent_next_states[agent_id][2] = (state_r, state_c - 1)
                if (state_r - 1 >= 0
                        and not((state_r - 1, state_c) in self.walls)): # up
                    agent_next_states[agent_id][1] = (state_r - 1, state_c) 
                if (state_r + 1 < self.Nr
                        and not((state_r + 1, state_c) in self.walls)): # down
                    agent_next_states[agent_id][3] = (state_r + 1, state_c) 
                agent_next_states[agent_id][4] = (state_r, state_c) # stay

                # action 0: move right
                # action 1: move up
                # action 2: move left
                # action 3: move down
                # action 4: stay

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

    def run_trajectory(self, policy : np.ndarray, max_steps : int = 30):
        """
        Run a trajectory from the joint initial state implementing the
        specified policy with full communication.

        Parameters
        ----------
        policy : 
            Matrix representing the policy. policy[s_ind, a_ind] is the 
            probability of taking the action indexed by a_ind from the 
            joint state indexed by s_ind.

        Returns
        -------
        traj : list
            List of indexes of states. 
        """
        traj = []
        traj.append(self.initial_index)
        s = self.initial_index

        while ((s not in self.target_indexes) and (s not in self.dead_indexes)
                    and len(traj) <= max_steps):
            a = np.random.choice(np.arange(self.Na_joint), p=policy[s,:])
            s = np.random.choice(np.arange(self.Ns_joint), p=self.T[s,a,:])
            traj.append(s)

        return traj

    def run_trajectory_imaginary(self, 
                                policy : np.ndarray, 
                                max_steps : int = 30):
        """
        Run a trajectory from the joint initial state under imaginary 
        play implementing the specified joint policy.

        Parameters
        ----------
        policy : 
            Matrix representing the policy. policy[s_ind, a_ind] is the 
            probability of taking the action indexed by a_ind from the 
            joint state indexed by s_ind.

        Returns
        -------
        traj : list
            List of indexes of states. 
        """
        # TODO: Make this code general for N agents.
        # TODO: Verify that this code actually implements fictitious play
        # how we want it to. I don't think the true next state should
        # just be the composition of each of the individual next states.
        traj = []
        s_tuple = self.pos_from_index[self.initial_index]
        s1_tuple = s_tuple
        s2_tuple = s_tuple
        s_tuple = s1_tuple[0:2] + s2_tuple[2:4]

        s1 = self.index_from_pos[s1_tuple]
        s2 = self.index_from_pos[s2_tuple]
        s = self.index_from_pos[s_tuple]
        traj.append(s)

        while ((s not in self.target_indexes) and (s not in self.dead_indexes)
                    and len(traj) <= max_steps):
            a1 = np.random.choice(np.arange(self.Na_joint), p=policy[s1,:])
            a2 = np.random.choice(np.arange(self.Na_joint), p=policy[s2,:])
            s1 = np.random.choice(np.arange(self.Ns_joint), p=self.T[s1,a1,:])
            s2 = np.random.choice(np.arange(self.Ns_joint), p=self.T[s2,a2,:])

            s1_tuple = self.pos_from_index[s1]
            s2_tuple = self.pos_from_index[s2]

            s_tuple = s1_tuple[0:2] + s2_tuple[2:4]
            s = self.index_from_pos[s_tuple]
            traj.append(s)

        return traj

        # traj = []
        # s_tuple = self.pos_from_index[self.initial_index]

        # agent_s_tuples = {}
        # agent_s_inds = {}
        # agent_a_inds = {}

        # for agent_id in range(self.N_agents):
        #     agent_s_tuples[agent_id] = s_tuple
        #     agent_s_inds[agent_id] = self.index_from_pos[agent_s_tuples[agent_id]]
        
        # for agent_id in range(self.N_agents):
        #     s_tuple = (s_tuple
        #                 + agent_s_tuples[agent_s_tuples][2*agent_id:(2*agent_id + 2)])

        # s = self.index_from_pos[s_tuple]
        # traj.append(s)

        # while ((s not in self.target_indexes) and (s not in self.dead_indexes)
        #             and len(traj) <= max_steps):
        #     a1 = np.random.choice(np.arange(self.Na_joint), p=policy[s1,:])
        #     a2 = np.random.choice(np.arange(self.Na_joint), p=policy[s2,:])
        #     s1 = np.random.choice(np.arange(self.Ns_joint), p=self.T[s1,a1,:])
        #     s2 = np.random.choice(np.arange(self.Ns_joint), p=self.T[s2,a2,:])

        #     s1_tuple = self.pos_from_index[s1]
        #     s2_tuple = self.pos_from_index[s2]

        #     s_tuple = s1_tuple[0:2] + s2_tuple[2:4]
        #     s = self.index_from_pos[s_tuple]
        #     traj.append(s)

        # return traj

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

        # Plot the initial states
        for agent_id in range(self.N_agents):
            (init_r, init_c) = self.initial_state[2*agent_id:(2*agent_id + 2)]
            # ax.text(init_c * grid_spacing + grid_spacing/2, 
            #         - (init_r * grid_spacing + grid_spacing/2), 
            #         'R{}'.format(agent_id))
            ax.plot(init_c * grid_spacing + grid_spacing/2, 
                    - (init_r * grid_spacing + grid_spacing/2), 
                    linestyle=None, marker='x', markersize=15, color='blue')

        # plot the current state
        if state is not None:
            for agent_id in range(self.N_agents):
                (state_r, state_c) = state[2*agent_id:(2*agent_id + 2)]
                ax.text(state_c * grid_spacing + grid_spacing/2, 
                    - (state_r * grid_spacing + grid_spacing/2), 
                    'R{}'.format(agent_id))
                # ax.plot(state_c * grid_spacing + grid_spacing/2,
                #         -(state_r * grid_spacing + grid_spacing/2), 
                #         linestyle=None, marker='o', markersize=20, color='blue')

        # Plot the target locations
        for goal in self.target_states:
            for agent_id in range(self.N_agents):
                (goal_r, goal_c) = goal[2*agent_id:(2*agent_id + 2)]
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

        # Plot the lava
        for lava in self.lava:
            (lava_r, lava_c) = lava
            lava_square = patches.Rectangle((lava_c * grid_spacing, 
                                            -(lava_r + 1) * grid_spacing), 
                                            grid_spacing, grid_spacing, 
                                            fill=True, color='red')
            ax.add_patch(lava_square)

        # # Plot the obstacles
        # for obstacle in self.dead_states:
        #     (obs_r, obs_c) = obstacle
        #     obs_square = patches.Rectangle((obs_c * grid_spacing, 
        #                                     -(obs_r + 1) * grid_spacing), 
        #                                     grid_spacing, grid_spacing, 
        #                                     fill=True, color='red')
        #     ax.add_patch(obs_square)

        if plot:
            plt.show()

    def create_trajectories_gif(self,
                                trajectories_list : list, 
                                save_folder_str : str,
                                save_file_name : str = 'ma_gridworld.gif'):
        """
        Create a gif illustrating a collection of trajectories in the
        multiagent environment.

        Parameters
        ----------
        trajectories_list : 
            A list of trajectories. Each trajectory is itself a list
            of the indexes of joint states.
        save_folder_str :
            The folder in which to save the gif.
        save_file_name :
            The desired name of the output file.
        """

        i = 0
        filenames = []
        for traj in trajectories_list:
            for state_ind in traj:
                state = self.pos_from_index[state_ind]

                # Create the plot of the current state
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(111, aspect='equal')
                self.display(state=state, ax=ax, plot=False)

                # Save the plot of the current state
                filename = os.path.join(save_folder_str, 'f{}.png'.format(i))
                filenames.append(filename)
                i = i + 1
                plt.savefig(filename)
                plt.close()
            
            # In between trajectories, add a blank screen.
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, aspect='equal')
            ax.axis('off')
            filename = os.path.join(save_folder_str, 'f{}.png'.format(i))
            filenames.append(filename)
            i = i + 1
            plt.savefig(filename)
            plt.close()

        im_list = []
        for filename in filenames:
            im_list.append(imageio.imread(filename))
        imageio.mimwrite(os.path.join(save_folder_str, save_file_name),
                            im_list,
                            duration=0.5)
        
        # Clean up the folder of all the saved pictures
        for filename in set(filenames):
            os.remove(filename)
        
def main():

    ##### BUILD THE GRIDWOLRD FROM SCRATCH

    # Build the gridworld
    print('Building gridworld')
    t_start = time.time()
    gridworld = MAGridworld(Nr=5, Nc=5, walls=[(0,2), (2,2), (4,2)])
    print('Constructed gridworld in {} seconds'.format(time.time() - t_start))

    # Sanity check on the transition matrix
    for s in range(gridworld.Ns_joint):
        for a in range(gridworld.Na_joint):
            assert(np.abs(np.sum(gridworld.T[s, a, :]) - 1.0) <= 1e-12)

    # Save the constructed gridworld
    save_file_str = os.path.join(os.path.abspath(os.path.curdir), 
                                    'saved_environments', 'ma_gridworld.pkl')
    gridworld.save(save_file_str)

    # ##### LOAD A PRE-CONSTRUCTED GRIDWORLD
    # load_file_str = os.path.join(os.path.abspath(os.path.curdir),
    #                                 'saved_environments', 'ma_gridworld.pkl')
    # gridworld = MAGridworld(load_file_str=load_file_str)
    # print('Loaded multiagent gridworld.')

    # ##### Examine the tradeoff between success probability and expected len
    # mdp = gridworld.build_mdp()
    # # Construct the reachability LP
    # prob, x = build_reachability_LP(mdp, exp_len_coef=0.0)
    # exp_len_coeff_list = np.linspace(0.0, 0.5, num=50)

    # succ_prob_list = []
    # exp_len_list = []

    # for exp_len_coeff_val in exp_len_coeff_list:
    #     prob.parameters()[0].value = exp_len_coeff_val
    #     prob.solve()
    #     # Get the optimal joint policy
    #     occupancy_vars = process_occupancy_vars(x)
    #     succ_prob_list.append(mdp.success_probability_from_occupancy_vars(occupancy_vars))
    #     exp_len_list.append(mdp.expected_len_from_occupancy_vars(occupancy_vars))

    # fig = plt.figure(figsize=(20,10))
    # ax = fig.add_subplot(211)
    # ax.plot(exp_len_coeff_list, succ_prob_list, 
    #         linewidth=3, marker='o', label='Success Probability')
    # ax.grid()
    # ax.set_xlabel('Expected Length Coefficient')
    # ax.set_ylabel('Success Probability')

    # ax = fig.add_subplot(212)
    # ax.plot(exp_len_coeff_list, exp_len_list, 
    #         linewidth=3, marker='o', label='Expected Length')
    # ax.grid()
    # ax.set_xlabel('Expected Length Coefficient')
    # ax.set_ylabel('Expected Length')

    # plt.show()

    ##### Solve and visualize the reachability problem for a reasonable value
    ##### of the expected length coefficient
    # Construct the corresponding MDP
    mdp = gridworld.build_mdp()

    # Construct and solve the reachability LP
    prob, x = build_reachability_LP(mdp, exp_len_coef=0.1)
    # prob.parameters()[0].value = 0.1
    prob.solve()

    # Get the optimal joint policy
    occupancy_vars = process_occupancy_vars(x)
    policy = mdp.policy_from_occupancy_vars(occupancy_vars)

    success_prob = mdp.success_probability_from_occupancy_vars(occupancy_vars)
    expected_len = mdp.expected_len_from_occupancy_vars(occupancy_vars)

    print('Success probability: {}, expected length: {}'.format(success_prob, 
                                                                expected_len))

    ##### Empirically measure imaginary play success
    num_trajectories = 10000
    success_count = 0
    for t_ind in range(num_trajectories):
        if gridworld.run_trajectory_imaginary(policy, max_steps=50)[-1] in gridworld.target_indexes:
            success_count = success_count + 1
    print('Imaginary play success rate: {}'.format(success_count / num_trajectories))

    ##### Create a GIF
    num_trajectories = 10
    trajectory_list = []
    for t_ind in range(num_trajectories):
        trajectory_list.append(gridworld.run_trajectory_imaginary(policy))

    gif_save_folder = os.path.join(os.path.abspath(os.path.curdir), 'gifs')

    gridworld.create_trajectories_gif(trajectory_list, gif_save_folder)

if __name__ == '__main__':
    main()