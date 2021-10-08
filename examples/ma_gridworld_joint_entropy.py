import matplotlib.pyplot as plt
import numpy as np
import sys, os, time

sys.path.append('../')

from environments.ma_gridworld import MAGridworld
from markov_decision_process.mdp import MDP
from optimization_problems.joint_entropy import build_joint_entropy_program
from utils.process_occupancy import *

# #### BUILD THE GRIDWOLRD FROM SCRATCH

# # Build the gridworld
# print('Building gridworld')
# t_start = time.time()
# gridworld = MAGridworld(N_agents=2,
#                         Nr=5, 
#                         Nc=5,
#                         slip_p=0.1,
#                         initial_state=(4, 0, 4, 4),
#                         target_states=[(4, 4, 4, 0)],
#                         dead_states=[],
#                         lava=[(0, 4), (0, 0)],
#                         walls=[(0,2), (2,2), (4,2)])

# print('Constructed gridworld in {} seconds'.format(time.time() - t_start))

# # Sanity check on the transition matrix
# for s in range(gridworld.Ns_joint):
#     for a in range(gridworld.Na_joint):
#         assert(np.abs(np.sum(gridworld.T[s, a, :]) - 1.0) <= 1e-12)

# # Save the constructed gridworld
# save_file_str = os.path.join(os.path.abspath(os.path.curdir),
#                                 '..', 'environments',
#                                 'saved_environments', 'ma_gridworld.pkl')
# gridworld.save(save_file_str)

##### LOAD A PRE-CONSTRUCTED GRIDWORLD
load_file_str = os.path.join(os.path.abspath(os.path.curdir),
                                '..', 'environments',
                                'saved_environments', 'ma_gridworld.pkl')
gridworld = MAGridworld(load_file_str=load_file_str)
print('Loaded multiagent gridworld.')

##### Solve and visualize the problem 
# Construct the corresponding MDP
mdp = gridworld.build_mdp()

# Construct and solve the reachability LP
t = time.time()
prob, vars, params = build_joint_entropy_program(mdp, 
                                                exp_len_coef=0.1,
                                                entropy_coef=0.05)
print('Constructed optimization problem in {} seconds.'.format(time.time() - t)) 

t = time.time()
prob.solve(verbose=True)
print('Solved the optimization problem in {} seconds'.format(time.time() - t)) 

# Get the optimal joint policy
occupancy_vars = process_occupancy_vars(vars[0])
policy = policy_from_occupancy_vars(mdp, occupancy_vars)

success_prob = success_probability_from_occupancy_vars(mdp, occupancy_vars)
expected_len = expected_len_from_occupancy_vars(mdp, occupancy_vars)
joint_entropy = compute_joint_entropy(mdp, occupancy_vars)

print(('Success probability: {}, \n \
        expected length: {}, \n \
        joint entropy: {}'.format(success_prob, 
                                    expected_len, 
                                    joint_entropy)))

##### Empirically measure imaginary play success
num_trajectories = 10000
success_count = 0
for t_ind in range(num_trajectories):
    if gridworld.run_trajectory(policy, max_steps=50)[-1] in gridworld.target_indexes:
        success_count = success_count + 1
print('Imaginary play success rate: {}'.format(success_count / num_trajectories))

##### Create a GIF
num_trajectories = 10
trajectory_list = []
for t_ind in range(num_trajectories):
    trajectory_list.append(gridworld.run_trajectory(policy, max_steps=50))

gif_save_folder = os.path.join(os.path.abspath(os.path.curdir), 'gifs')

gridworld.create_trajectories_gif(trajectory_list, 
                                    gif_save_folder,
                                    save_file_name='ma_gridworld_joint_entropy.gif')