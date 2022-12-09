# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import os, sys, time
sys.path.append('..')

import tikzplotlib
from matplotlib import cm

from environments.ma_gridworld import MAGridworld
from utils.experiment_logger import ExperimentLogger

# %%
# Plotting parameters
fontsize = 12

water_color = '#586494'
mountain_color = '#8c8188'
target_color = '#4472C4'

tikz_save_path = os.path.abspath(os.path.join(os.path.curdir, 'tikz'))

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
# save_file_name = '2021-09-25-19-44-09_ma_gridworld_total_corr_slip_0p05.pkl'
# save_file_name = '2021-10-06-15-51-09_ma_gridworld_total_corr_slip_0p05.pkl'
save_file_name = '2022-01-06-12-23-20_ma_gridworld_total_corr_add_end_state_0p05.pkl'
save_str = os.path.join(base_path, save_file_name)

exp_logger = ExperimentLogger(load_file_str=save_str)

# save_file_name_reachability = '2021-09-25-20-05-29_ma_gridworld_reachability_slip_0p05.pkl'
save_file_name_reachability = '2021-10-06-16-09-44_ma_gridworld_reachability_0p05.pkl'
save_str_reachability = os.path.join(base_path, save_file_name_reachability)
exp_logger_reachability = ExperimentLogger(load_file_str=save_str_reachability)

# %%
# Construct the gridworld to get the state mappings
env = MAGridworld(**exp_logger.environment_settings)

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

agent_id = 1
plot_reachability = True

for r in range(Nr):
    for c in range(Nc):
        s_ind_list = []
        for s_ind in range(N_joint_states):
            pos = env.pos_from_index[s_ind][2*agent_id:(2*agent_id+2)]
            if pos == (r, c):
                s_ind_list.append(s_ind)
        if plot_reachability:
            agent0_map_total_corr[r, c] = np.sum(x_reachability[s_ind_list])
        else:
            agent0_map_total_corr[r, c] = np.sum(x_total_corr[s_ind_list])

goal_local_pos = env.target_states[0][2*agent_id:(2*agent_id+2)]
agent0_map_total_corr[goal_local_pos[0], goal_local_pos[1]] = 1.0

# %%
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.axis('off')

print(np.max(agent0_map_total_corr))

# 2.83
# 1.05
# 1.39
# 3.26

# 3.26 was the max value observed. Hardcoding it to allow for a comparable heatmap across all plots.
heatmap = ax.imshow(agent0_map_total_corr, cmap=plt.get_cmap('Oranges'), 
                                                    interpolation='nearest', 
                                                    vmin=0, vmax=3.26) 

# walls
ax.add_patch(Rectangle([1.5, 3.5], 1.0, 1.0, edgecolor='black', 
                                                facecolor=mountain_color))
ax.add_patch(Rectangle([1.5, 1.5], 1.0, 1.0, edgecolor='black',
                                                facecolor=mountain_color))
ax.add_patch(Rectangle([1.5, -0.5], 1.0, 1.0, edgecolor='black', 
                                                facecolor=mountain_color))
# ax.add_patch(Rectangle)

# Water
ax.add_patch(Rectangle([0.5, -0.5], 1.0, 1.0, edgecolor='black', 
                                                facecolor=water_color))
ax.add_patch(Rectangle([-0.5, -0.5], 1.0, 1.0, edgecolor='black', 
                                                facecolor=water_color))
ax.add_patch(Rectangle([2.5, -0.5], 1.0, 1.0, edgecolor='black', 
                                                facecolor=water_color))

# Plot a rectangle around the whole thing.
ax.add_patch(Rectangle([-0.5, -0.5], 5.0, 5.0, facecolor='none',
                                                edgecolor='black',
                                                linewidth=8))

ax.plot([0.0, 0.0], color=water_color)

# # Use this when agent_id = 0
# ax.text(0.0, 4.0, '$R_2$\n Initial State', fontsize=18, color='white',
#                                             horizontalalignment='center')
# ax.text(3.0, 4.0, '$T_2$', fontsize=20, color='white',
#                                             horizontalalignment='center')

# # Use this when agent_id = 1
# ax.text(4.0, 4.0, '$R_1$\n Initial State', fontsize=18, color='white',
#                                             horizontalalignment='center')
# ax.text(1.0, 4.0, '$T_1$', fontsize=20, color='white',
#                                             horizontalalignment='center')

# cbar = plt.colorbar(heatmap)
# # ticklabs = cbar.ax.get_yticklabels()
# # cbar.ax.set_yticklabels(ticklabs, fontsize=150)
# for t in cbar.ax.get_yticklabels():
#      t.set_fontsize(20)

# plt.show()

plt.savefig('agent1_map_total_reachability.pdf', bbox_inches='tight')


# %%
