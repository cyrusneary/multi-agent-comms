import matplotlib.pyplot as plt
import numpy as np

import os, sys, time
sys.path.append('..')

import tikzplotlib

from environments.ma_gridworld import MAGridworld
from utils.experiment_logger import ExperimentLogger

# Plotting parameters
fontsize = 12

tikz_save_path = os.path.abspath(os.path.join(os.path.curdir, 'tikz'))

##### Load the policies

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
save_file_name = '2021-10-06-15-51-09_ma_gridworld_total_corr_slip_0p05.pkl' # data used in initial submission
save_file_name = '2022-01-06-12-23-20_ma_gridworld_total_corr_add_end_state_0p05.pkl' # data used in final submission
save_str = os.path.join(base_path, save_file_name)

exp_logger = ExperimentLogger(load_file_str=save_str)

save_file_name_reachability = '2021-09-25-20-05-29_ma_gridworld_reachability_slip_0p05.pkl'
save_str_reachability = os.path.join(base_path, save_file_name_reachability)
exp_logger_reachability = ExperimentLogger(load_file_str=save_str_reachability)

##### Create the gridworld from the logged parameters

t_start = time.time()
gridworld = MAGridworld(**exp_logger.environment_settings)
print('Constructed the gridworld in {} seconds.'.format(time.time() - t_start))

q_list = np.linspace(0, 1.0, num=11)
md_success_probs = []
base_success_probs = []
lower_bound_list = []

for q in q_list:
    md_policy = exp_logger.results[max(exp_logger.results.keys())]['policy']
    base_policy = exp_logger_reachability.results[max(exp_logger_reachability.results.keys())]['policy']

    md_policy_val = exp_logger.results[max(exp_logger.results.keys())]['success_prob']
    md_total_corr = exp_logger.results[max(exp_logger.results.keys())]['total_corr']

    md_success_prob = \
        gridworld.empirical_intermittent_success_rate(md_policy, q,
                                            num_trajectories=10000,
                                            max_steps_per_trajectory=200)
    base_success_prob = \
        gridworld.empirical_intermittent_success_rate(base_policy, q,
                                            num_trajectories=10000,
                                            max_steps_per_trajectory=200)
    lower_bound = md_policy_val - np.sqrt(1 - np.exp(-q * md_total_corr))

    print('Finished simulating q = {}'.format(q))

    md_success_probs.append(md_success_prob)
    base_success_probs.append(base_success_prob)
    lower_bound_list.append(lower_bound)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(q_list, md_success_probs, color='blue')
ax.plot(q_list, base_success_probs, color='red')
ax.plot(q_list, lower_bound_list, color='black')

ax.grid()
ax.set_xlabel('q', fontsize=fontsize)
ax.set_ylabel('Task Success Probability', fontsize=fontsize)

tikz_file_str = os.path.join(tikz_save_path, 'plot_intermittent_play_data_aux_action.tex')
tikzplotlib.save(tikz_file_str)

plt.show()