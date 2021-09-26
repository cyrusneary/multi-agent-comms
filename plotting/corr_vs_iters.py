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

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
save_file_name = '2021-09-20-17-10-46_ma_gridworld_total_corr_slip_0p05.pkl'
save_str = os.path.join(base_path, save_file_name)

exp_logger = ExperimentLogger(load_file_str=save_str)

save_file_name_reachability = '2021-09-24-16-44-35_ma_gridworld_reachability_slip_0p05.pkl'
save_str_reachability = os.path.join(base_path, save_file_name_reachability)
exp_logger_reachability = ExperimentLogger(load_file_str=save_str_reachability)
success_prob_reachability = exp_logger_reachability.results[0]['success_prob'] * np.ones((99,))
empirical_imag_reachability = exp_logger_reachability.results[0]['empirical_imag_success_rate'] * np.ones((99,))

# Get the relevant data in numpy format
iters_indexes = exp_logger.results.keys()

total_corr = []
success_prob = []
empirical_imag = []
iters = []
for key in range(max(iters_indexes)):
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
ax.plot(iters, exp_logger_reachability.results[0]['total_corr'] * np.ones(len(iters)), color='magenta')
ax.grid()
ax.set_ylabel('Total Correlation Value', fontsize=fontsize)
ax.set_xlabel('Number of Convex-Concave Iterations', fontsize=fontsize)
ax.set_title('Total Correlation During Policy Synthesis', fontsize=fontsize)
ax.set_yscale('log')

tikz_file_str = os.path.join(tikz_save_path, 'total_corr_vs_iters.tex')
# tikzplotlib.save(tikz_file_str)

plt.show()

# Plot 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(iters, success_prob, 
            color='green', marker='.', label='Success Probability [Full Comms]')
ax.plot(iters, success_prob_reachability,
            color='green', linestyle='--', 
            label='Success Probability [Full Comms], \n Optimal Reachability Policy')
ax.plot(iters, bound,
            color='black', 
            label='Theoretical Lower Bound on Success Probability of Imaginary Play')
ax.plot(iters, empirical_imag,
            color='magenta', marker='.', 
            label='Empirically Measured Success Probability of Imaginary Play')
ax.plot(iters, empirical_imag_reachability,
            color='magenta', linestyle='--',
            label='Empirically Measured Success Probability of Imaginary Play \n Optimal Reachability Policy')
ax.grid()
ax.set_ylabel('Total Correlation Value', fontsize=fontsize)
ax.set_xlabel('Number of Convex-Concave Iterations', fontsize=fontsize)
ax.set_title('Policy Success Probability During Synthesis', fontsize=fontsize)
plt.legend(fontsize=fontsize)

tikz_file_str = os.path.join(tikz_save_path, 'success_prob_vs_iters.tex')
# tikzplotlib.save(tikz_file_str)

plt.show()