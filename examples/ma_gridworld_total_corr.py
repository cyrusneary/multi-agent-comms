from cvxpy.error import SolverError
import matplotlib.pyplot as plt
import numpy as np
import sys, os, time

sys.path.append('../')

from environments.ma_gridworld import MAGridworld
from markov_decision_process.mdp import MDP
from optimization_problems.total_corr2 import build_linearized_program,\
                                                    compute_total_correlation
from optimization_problems.joint_entropy import build_joint_entropy_program
from optimization_problems.reachability_LP import build_reachability_LP
from optimization_problems.random_policy_program import build_random_policy_program

from utils.process_occupancy import *

rebuild_gridworld = False

### BUILD THE GRIDWOLRD FROM SCRATCH
if rebuild_gridworld:
    # Build the gridworld
    print('Building gridworld')
    t_start = time.time()
    gridworld = MAGridworld(N_agents=2,
                            Nr=5, 
                            Nc=5,
                            slip_p=0.05, # 0.1
                            initial_state=(4, 0, 4, 4),
                            target_states=[(4, 4, 4, 0)],
                            dead_states=[],
                            lava=[(0, 0), (0,3), (0,1)], #(0,0), (0,3), (0,1)
                            walls=[(0,2), (2,2), (4,2)])

    print('Constructed gridworld in {} seconds'.format(time.time() - t_start))

    # Sanity check on the transition matrix
    for s in range(gridworld.Ns_joint):
        for a in range(gridworld.Na_joint):
            assert(np.abs(np.sum(gridworld.T[s, a, :]) - 1.0) <= 1e-12)

    # Save the constructed gridworld
    save_file_str = os.path.join(os.path.abspath(os.path.curdir),
                                    '..', 'environments',
                                    'saved_environments', 'ma_gridworld.pkl')
    gridworld.save(save_file_str)

##### LOAD A PRE-CONSTRUCTED GRIDWORLD
else:
    load_file_str = os.path.join(os.path.abspath(os.path.curdir),
                                    '..', 'environments',
                                    'saved_environments', 'ma_gridworld.pkl')
    gridworld = MAGridworld(load_file_str=load_file_str)
    print('Loaded multiagent gridworld.')

##### Start with an easier optimization problem to get an initial guess.

# Construct the corresponding MDP
mdp = gridworld.build_mdp()

# t = time.time()
# prob_init, vars_init, _ = build_joint_entropy_program(mdp,
#                                                 exp_len_coef=0.0,
#                                                 entropy_coef=0.01)
# print('Constructed optimization problem in {} seconds.'.format(time.time() - t)) 

# # Construct and solve the reachability LP
# prob_init, vars_init, params_init = build_reachability_LP(mdp, 
#                                                             exp_len_coef=0.01)

# Construct and solve for a random initial policy
rand_init = np.random.rand(mdp.Ns, mdp.Na)
prob_init, vars_init, params_init = build_random_policy_program(mdp, 
                                                        rand_init=rand_init)
##### Build the linearized optimization problem

agent_state_size_list = []
agent_action_size_list = []
for agent_id in range(gridworld.N_agents):
    agent_state_size_list.append(gridworld.Ns_local)
    agent_action_size_list.append(gridworld.Na_local)

# Construct and solve the reachability LP
t = time.time()
prob, vars, params, f_grad, g_grad = build_linearized_program(mdp, 
                                            gridworld.N_agents,
                                            agent_state_size_list,
                                            agent_action_size_list,
                                            gridworld.check_agent_state_action,
                                            gridworld.check_agent_state,
                                            reachability_coef=10.0, # 10.0
                                            exp_len_coef=0.1, #0.05
                                            total_corr_coef=4.0) #1.0
print('Constructed optimization problem in {} seconds.'.format(time.time()-t)) 

##### Solve the initial problem to get an initial guess

t = time.time()
prob_init.solve(verbose=True, solver='ECOS')
print('Solved the entropy maximization \
            optimization problem in {} seconds'.format(time.time() - t)) 

occupancy_vars_start = process_occupancy_vars(vars_init[0])
success_prob = success_probability_from_occupancy_vars(mdp, occupancy_vars_start)
expected_len = expected_len_from_occupancy_vars(mdp, occupancy_vars_start)
joint_entropy = compute_joint_entropy(mdp, occupancy_vars_start)
total_corr = compute_total_correlation(mdp,
                            N_agents=gridworld.N_agents,
                            agent_state_size_list=agent_state_size_list,
                            agent_action_size_list=agent_action_size_list,
                            f_grad=f_grad,
                            g_grad=g_grad,
                            x=occupancy_vars_start)
print(('Success probability: {}, \n \
        expected length: {}, \n \
        joint entropy: {}\n \
        total correlation: {}'.format(success_prob, 
                                    expected_len, 
                                    joint_entropy, 
                                    total_corr)))

policy_start = policy_from_occupancy_vars(mdp, occupancy_vars_start)
x_start = occupancy_vars_start

##### Solve the problem via convex-concave procedure

ccp_tol = 1e-6
x_last = x_start
opt_val_list = []
success_prob_list = []
expected_len_list = []
joint_entropy_list = []
total_correlation_list = []
for i in range(100):
    params[3].value = x_last
    prob.solve()
    opt_val_list.append(prob.solution.opt_val)

    occupancy_vars = process_occupancy_vars(vars[0])

    x_last = occupancy_vars

    total_correlation_list.append(compute_total_correlation(mdp,
                                N_agents=gridworld.N_agents,
                                agent_state_size_list=agent_state_size_list,
                                agent_action_size_list=agent_action_size_list,
                                f_grad=f_grad,
                                g_grad=g_grad,
                                x=occupancy_vars))
    success_prob = success_probability_from_occupancy_vars(mdp, occupancy_vars)
    print('[{}]: Success probability: {}, \
            total correlation: {}'.format(i, success_prob, 
                                            total_correlation_list[-1]))
    success_prob_list.append(success_prob)
    expected_len_list.append(expected_len_from_occupancy_vars(mdp, occupancy_vars))
    joint_entropy_list.append(compute_joint_entropy(mdp, occupancy_vars))

policy = policy_from_occupancy_vars(mdp, occupancy_vars)

print(opt_val_list)
print(total_correlation_list)
print(success_prob_list)
print(expected_len_list)
# print(joint_entropy_list)

##### Empirically measure imaginary play success
num_trajectories = 10000
success_count = 0
for t_ind in range(num_trajectories):
    if gridworld.run_trajectory_imaginary(policy, max_steps=50)[-1] in gridworld.target_indexes:
        success_count = success_count + 1
print('Imaginary play success rate: {}'.format(success_count / num_trajectories))

##### Create a GIF

# First create a gif of the initial max entropy policy.
num_trajectories = 5
trajectory_list = []
for t_ind in range(num_trajectories):
    trajectory_list.append(gridworld.run_trajectory(policy_start, max_steps=50))

gif_save_folder = os.path.join(os.path.abspath(os.path.curdir), 'gifs')

gridworld.create_trajectories_gif(trajectory_list, 
                                    gif_save_folder,
                                    save_file_name='ma_gridworld_joint_entropy.gif')

# Next create a gif of the end policy.
num_trajectories = 5
trajectory_list = []
for t_ind in range(num_trajectories):
    trajectory_list.append(gridworld.run_trajectory(policy, max_steps=50))

gif_save_folder = os.path.join(os.path.abspath(os.path.curdir), 'gifs')

gridworld.create_trajectories_gif(trajectory_list, 
                                    gif_save_folder,
                                    save_file_name='ma_gridworld_total_corr.gif')