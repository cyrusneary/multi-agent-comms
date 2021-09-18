import matplotlib.pyplot as plt
import numpy as np
import sys, os, time

sys.path.append('../')

from environments.ma_gridworld import MAGridworld
from markov_decision_process.mdp import MDP
from optimization_problems.reachability_LP import build_reachability_LP,\
                                                    process_occupancy_vars

#### BUILD THE GRIDWOLRD FROM SCRATCH

# Build the gridworld
print('Building gridworld')
t_start = time.time()
gridworld = MAGridworld(N_agents=2,
                        Nr=5, 
                        Nc=5,
                        slip_p=0.1,
                        initial_state=(4, 0, 4, 4),
                        target_states=[(4, 4, 4, 0)],
                        dead_states=[],
                        lava=[(0, 4), (0, 0)],
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

##### Examine the tradeoff between success probability and expected len
mdp = gridworld.build_mdp()
# Construct the reachability LP
prob, vars, params = build_reachability_LP(mdp, exp_len_coef=0.0)
exp_len_coeff_list = np.linspace(0.0, 0.5, num=50)

succ_prob_list = []
exp_len_list = []

for exp_len_coeff_val in exp_len_coeff_list:
    params[0].value = exp_len_coeff_val
    prob.solve()
    # Get the optimal joint policy
    occupancy_vars = process_occupancy_vars(vars[0])
    succ_prob_list.append(mdp.success_probability_from_occupancy_vars(occupancy_vars))
    exp_len_list.append(mdp.expected_len_from_occupancy_vars(occupancy_vars))

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(exp_len_coeff_list, succ_prob_list, 
        linewidth=3, marker='o', label='Success Probability')
ax.grid()
ax.set_xlabel('Expected Length Coefficient')
ax.set_ylabel('Success Probability')

ax = fig.add_subplot(212)
ax.plot(exp_len_coeff_list, exp_len_list, 
        linewidth=3, marker='o', label='Expected Length')
ax.grid()
ax.set_xlabel('Expected Length Coefficient')
ax.set_ylabel('Expected Length')

plt.show()

##### Solve and visualize the reachability problem for a reasonable value
##### of the expected length coefficient
# Construct the corresponding MDP
mdp = gridworld.build_mdp()

# Construct and solve the reachability LP
prob, vars, params = build_reachability_LP(mdp, exp_len_coef=0.1)
prob.solve()

# Get the optimal joint policy
occupancy_vars = process_occupancy_vars(vars[0])
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
    trajectory_list.append(gridworld.run_trajectory_imaginary(policy, max_steps=50))

gif_save_folder = os.path.join(os.path.abspath(os.path.curdir), 'gifs')

gridworld.create_trajectories_gif(trajectory_list, gif_save_folder)