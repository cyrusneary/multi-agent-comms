import matplotlib.pyplot as plt
import numpy as np
import sys, os, time

sys.path.append('../')

from environments.ma_gridworld import MAGridworld
from markov_decision_process.mdp import MDP
from optimization_problems.total_corr2 import build_linearized_program,\
                                                    process_occupancy_vars,\
                                                        compute_total_correlation
from optimization_problems.joint_entropy import build_joint_entropy_program
from optimization_problems.reachability_LP import build_reachability_LP

### BUILD THE GRIDWOLRD FROM SCRATCH

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

# ##### LOAD A PRE-CONSTRUCTED GRIDWORLD
# load_file_str = os.path.join(os.path.abspath(os.path.curdir),
#                                 '..', 'environments',
#                                 'saved_environments', 'ma_gridworld.pkl')
# gridworld = MAGridworld(load_file_str=load_file_str)
# print('Loaded multiagent gridworld.')

##### Start by solving the entropy maximization problem to get initial guess
# Construct the corresponding MDP
mdp = gridworld.build_mdp()

# t = time.time()
# prob_init, vars_init, _ = build_joint_entropy_program(mdp,
#                                                 exp_len_coef=0.0,
#                                                 entropy_coef=0.01)
# print('Constructed optimization problem in {} seconds.'.format(time.time() - t)) 

# Construct and solve the reachability LP
prob_init, vars_init, params_init = build_reachability_LP(mdp, 
                                                            exp_len_coef=0.01)
prob_init.solve()

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
                                            exp_len_coef=0.03, #0.05
                                            total_corr_coef=4.0) #1.0
print('Constructed optimization problem in {} seconds.'.format(time.time()-t)) 

t = time.time()
prob_init.solve(verbose=True)
print('Solved the entropy maximization \
            optimization problem in {} seconds'.format(time.time() - t)) 

occupancy_vars = process_occupancy_vars(vars_init[0])
success_prob = mdp.success_probability_from_occupancy_vars(occupancy_vars)
expected_len = mdp.expected_len_from_occupancy_vars(occupancy_vars)
joint_entropy = mdp.compute_joint_entropy(occupancy_vars)
total_corr = compute_total_correlation(mdp,
                            N_agents=gridworld.N_agents,
                            agent_state_size_list=agent_state_size_list,
                            agent_action_size_list=agent_action_size_list,
                            f_grad=f_grad,
                            g_grad=g_grad,
                            x=occupancy_vars)
print(('Success probability: {}, \n \
        expected length: {}, \n \
        joint entropy: {}\n \
        total correlation: {}'.format(success_prob, 
                                    expected_len, 
                                    joint_entropy, 
                                    total_corr)))

policy_start = mdp.policy_from_occupancy_vars(occupancy_vars)
x_start = vars_init[0].value

x_start[np.where(x_start <= 1e-10)] = 1e-10

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
    x_last = vars[0].value
    x_last[np.where(x_last <= 1e-10)] = 1e-10

    occupancy_vars = process_occupancy_vars(vars[0])

    total_correlation_list.append(compute_total_correlation(mdp,
                                N_agents=gridworld.N_agents,
                                agent_state_size_list=agent_state_size_list,
                                agent_action_size_list=agent_action_size_list,
                                f_grad=f_grad,
                                g_grad=g_grad,
                                x=occupancy_vars))
    success_prob = mdp.success_probability_from_occupancy_vars(occupancy_vars)
    print('[{}]: Success probability: {}, \
            total correlation: {}'.format(i, success_prob, total_correlation_list[-1]))
    success_prob_list.append(success_prob)
    expected_len_list.append(mdp.expected_len_from_occupancy_vars(occupancy_vars))
    joint_entropy_list.append(mdp.compute_joint_entropy(occupancy_vars))

policy = mdp.policy_from_occupancy_vars(occupancy_vars)

print(opt_val_list)
print(total_correlation_list)
print(success_prob_list)
print(expected_len_list)
# print(joint_entropy_list)

# ##### Examine the tradeoff between success probability and expected len
# mdp = gridworld.build_mdp()
# # Construct the reachability LP
# prob, x = build_joint_entropy_program(mdp, 
#                                         exp_len_coef=0.0,
#                                         entropy_coef=0.0)
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

# fig = plt.figure()
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

# ##### Solve and visualize the reachability problem for a reasonable value
# ##### of the expected length coefficient
# # Construct the corresponding MDP
# mdp = gridworld.build_mdp()

# # Construct and solve the reachability LP
# t = time.time()
# prob, vars, params = build_joint_entropy_program(mdp, 
#                                                 exp_len_coef=0.1,
#                                                 entropy_coef=0.05)
# print('Constructed optimization problem in {} seconds.'.format(time.time() - t)) 

# t = time.time()
# prob.solve(verbose=True)
# print('Solved the entropy maximization optimization problem in {} seconds'.format(time.time() - t)) 

# # Get the optimal joint policy
# occupancy_vars = process_occupancy_vars(vars[0])
# policy = mdp.policy_from_occupancy_vars(occupancy_vars)

# success_prob = mdp.success_probability_from_occupancy_vars(occupancy_vars)
# expected_len = mdp.expected_len_from_occupancy_vars(occupancy_vars)
# joint_entropy = mdp.compute_joint_entropy(occupancy_vars)

# print(('Success probability: {}, \n \
#         expected length: {}, \n \
#         joint entropy: {}'.format(success_prob, 
#                                     expected_len, 
#                                     joint_entropy)))

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