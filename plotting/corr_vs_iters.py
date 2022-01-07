import matplotlib.pyplot as plt
import numpy as np

import os, sys, time
sys.path.append('..')

import tikzplotlib

from utils.experiment_logger import ExperimentLogger

# Plotting parameters
fontsize = 12
num_data_points = 51

tikz_save_path = os.path.abspath(os.path.join(os.path.curdir, 'tikz'))

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
# save_file_name = '2021-10-06-15-51-09_ma_gridworld_total_corr_slip_0p05.pkl' # data used in initial submission
# save_file_name = '2021-10-08-14-43-52_three_agent_gridworld_total_corr_0p05.pkl' # three agent example from initial submission
# save_file_name = '2022-01-06-12-23-20_ma_gridworld_total_corr_add_end_state_0p05.pkl' # data used in final submission
save_file_name = '2022-01-06-14-17-07_three_agent_gridworld_total_corr_0p05.pkl' # three agent example from final submission
save_str = os.path.join(base_path, save_file_name)

exp_logger = ExperimentLogger(load_file_str=save_str)

# save_file_name_reachability = '2021-10-06-16-09-44_ma_gridworld_reachability_0p05.pkl' # data used in paper
save_file_name_reachability = '2021-10-08-14-40-21_three_agent_gridworld_reachability_0p05.pkl' # three agent example in paper
save_str_reachability = os.path.join(base_path, save_file_name_reachability)
exp_logger_reachability = ExperimentLogger(load_file_str=save_str_reachability)
success_prob_reachability = exp_logger_reachability.results[0]['success_prob'] * np.ones((num_data_points,))
empirical_imag_reachability = exp_logger_reachability.results[0]['empirical_imag_success_rate'] * np.ones((num_data_points,))

# Get the relevant data in numpy format
iters_indexes = exp_logger.results.keys()

total_corr = []
success_prob = []
empirical_imag = []
iters = []
for key in range(num_data_points):
    total_corr.append(exp_logger.results[key]['total_corr'])
    success_prob.append(exp_logger.results[key]['success_prob'])
    empirical_imag.append(exp_logger.results[key]['empirical_imag_success_rate'])
    iters.append(key)

bound = np.array(success_prob) - np.sqrt(1 - np.exp(-np.array(total_corr)))

# Plot the total correlation as a function of optimization iterations
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(iters, total_corr,
        color='blue', marker='.')
# ax.plot(iters, exp_logger_reachability.results[0]['total_corr'] * np.ones(len(iters)), color='magenta')
ax.grid()
ax.set_ylabel('Total Correlation Value', fontsize=fontsize)
ax.set_xlabel('Number of Convex-Concave Iterations', fontsize=fontsize)
ax.set_title('Total Correlation During Policy Synthesis', fontsize=fontsize)

tikz_file_str = os.path.join(tikz_save_path, 'total_corr_vs_iters_three_agent_aux_action.tex')
tikzplotlib.save(tikz_file_str)

plt.show()

# Plot 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(iters, success_prob, 
            color='blue', marker='.', label='Success Probability [Full Comms]')
ax.plot(iters, success_prob_reachability,
            color='red', marker='.', 
            label='Success Probability [Full Comms], \n Optimal Reachability Policy')
ax.plot(iters, bound,
            color='black', 
            label='Theoretical Lower Bound on Success Probability of Imaginary Play')
ax.plot(iters, empirical_imag,
            color='blue', linestyle='--', 
            label='Empirically Measured Success Probability of Imaginary Play')
ax.plot(iters, empirical_imag_reachability,
            color='red', linestyle='--',
            label='Empirically Measured Success Probability of Imaginary Play \n Optimal Reachability Policy')
ax.grid()
ax.set_ylabel('Total Correlation Value', fontsize=fontsize)
ax.set_xlabel('Number of Convex-Concave Iterations', fontsize=fontsize)
ax.set_title('Policy Success Probability During Synthesis', fontsize=fontsize)
plt.legend(fontsize=fontsize)

tikz_file_str = os.path.join(tikz_save_path, 'success_prob_vs_iters_three_agent_aux_action.tex')
tikzplotlib.save(tikz_file_str)

plt.show()