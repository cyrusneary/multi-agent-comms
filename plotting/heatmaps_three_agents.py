# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import os, sys, time
sys.path.append('..')

import tikzplotlib
from matplotlib import cm

from environments.three_agent_gridworld import ThreeAgentGridworld
from utils.experiment_logger import ExperimentLogger

# %%
# Plotting parameters
fontsize = 12

water_color = '#586494'
mountain_color = '#8c8188'
target_color = '#4472C4'

tikz_save_path = os.path.abspath(os.path.join(os.path.curdir, 'tikz'))

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
# save_file_name = '2021-10-08-14-43-52_three_agent_gridworld_total_corr_0p05.pkl'
save_file_name = '2022-01-06-14-17-07_three_agent_gridworld_total_corr_0p05.pkl' # data file used in final submission
save_str = os.path.join(base_path, save_file_name)

exp_logger = ExperimentLogger(load_file_str=save_str)

save_file_name_reachability = '2021-10-08-14-40-21_three_agent_gridworld_reachability_0p05.pkl'
save_str_reachability = os.path.join(base_path, save_file_name_reachability)
exp_logger_reachability = ExperimentLogger(load_file_str=save_str_reachability)

# %%
# Construct the gridworld to get the state mappings
t_start = time.time()
load_file_str = os.path.join(os.path.abspath(os.path.curdir),
                                    '..', 'environments',
                                    'saved_environments', 'three_agent_gridworld.pkl')
env = ThreeAgentGridworld(load_file_str=load_file_str)

# %%
# Get the occupancy measures

# Get the relevant data in numpy format
iters_indexes = exp_logger.results.keys()
x_total_corr = exp_logger.results[max(iters_indexes)]['occupancy_vars']
x_reachability = exp_logger_reachability.results[0]['occupancy_vars']

# %%
# Construct the occupancy vars maps
Nr = exp_logger.environment_settings['Nr']
Nc = exp_logger.environment_settings['Nc']
N_joint_states = x_total_corr.shape[0]

agent0_map_total_corr = np.zeros((Nr, Nc))

agent_id = 0

for r in range(Nr):
    for c in range(Nc):
        s_ind_list = []
        for s_ind in range(N_joint_states):
            pos = env.pos_from_index[s_ind][2*agent_id:(2*agent_id+2)]
            if pos == (r, c):
                s_ind_list.append(s_ind)
        agent0_map_total_corr[r, c] = np.sum(x_total_corr[s_ind_list])

goal_local_pos = env.target_states[0][2*agent_id:(2*agent_id+2)]
agent0_map_total_corr[goal_local_pos[0], goal_local_pos[1]] = 1.0

# %%
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.axis('off')

heatmap = ax.imshow(agent0_map_total_corr, cmap=plt.get_cmap('Oranges'), 
                                                    interpolation='nearest')

# Plot a rectangle around the whole thing.
ax.add_patch(Rectangle([-0.5, -0.5], 3.0, 3.0, facecolor='none',
                                                edgecolor='black',
                                                linewidth=8))

# ax.plot([0.0, 0.0], color=water_color)

# agent_id = 0
ax.text(2.0, 2.0, '$R_1$\n Initial State', fontsize=27, color='white',
                                            horizontalalignment='center')
ax.text(0.0, 0.0, '$T_1$', fontsize=40, color='white',
                                            horizontalalignment='center')

# # agent_id = 1
# ax.text(0.0, 0.0, '$R_2$\n Initial State', fontsize=27, color='white',
#                                             horizontalalignment='center')
# ax.text(2.0, 2.0, '$T_2$', fontsize=40, color='white',
#                                             horizontalalignment='center')

# # agent_id = 2
# ax.text(2.0, 0.0, '$R_3$\n Initial State', fontsize=27, color='white',
#                                             horizontalalignment='center')
# ax.text(0.0, 2.0, '$T_3$', fontsize=40, color='white',
#                                             horizontalalignment='center')

cbar = plt.colorbar(heatmap)
# ticklabs = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(ticklabs, fontsize=150)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)

plt.show()


# %%
