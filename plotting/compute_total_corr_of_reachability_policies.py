
# %%
from cvxpy.error import SolverError
import numpy as np
import datetime
import sys, os, time

sys.path.append('../')

from environments.ma_gridworld import MAGridworld

# from optimization_problems.total_corr2 import build_linearized_program
from optimization_problems.total_corr_add_end_state import *
from optimization_problems.joint_entropy_add_aux_action import build_joint_entropy_program
from optimization_problems.reachability_LP import build_reachability_LP
from optimization_problems.random_policy_program_add_aux_action import build_random_policy_program

from utils.experiment_logger import ExperimentLogger
# from utils.process_occupancy import *

##### Solve the problem via convex-concave procedure
base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
save_file_name_reachability = '2021-10-06-16-09-44_ma_gridworld_reachability_0p05.pkl' # data used in paper
# save_file_name_reachability = '2021-10-08-14-40-21_three_agent_gridworld_reachability_0p05.pkl' # three agent example in paper
save_str_reachability = os.path.join(base_path, save_file_name_reachability)
exp_logger_reachability = ExperimentLogger(load_file_str=save_str_reachability)

# %%

t_start = time.time()
gridworld = MAGridworld(**exp_logger_reachability.environment_settings)
print('Constructed the gridworld in {} seconds.'.format(time.time() - t_start))

# Construct the corresponding MDP
mdp = gridworld.build_mdp()

agent_state_size_list = []
agent_action_size_list = []
for agent_id in range(gridworld.N_agents):
    agent_state_size_list.append(gridworld.Ns_local)
    agent_action_size_list.append(gridworld.Na_local)

# Construct and solve the optimization problem
t = time.time()
prob, vars, params, f_grad, g_grad = \
    build_linearized_program(mdp, 
        gridworld.N_agents,
        agent_state_size_list,
        agent_action_size_list,
        gridworld.check_agent_state_action_with_aux,
        gridworld.check_agent_state,
        reachability_coef=exp_logger_reachability.optimization_params['reachability_coef'],
        exp_len_coef=exp_logger_reachability.optimization_params['exp_len_coef'],
        total_corr_coef=exp_logger_reachability.optimization_params['total_corr_coef'])
print('Constructed optimization problem in {} seconds.'.format(time.time()-t)) 

# %%

total_corr = compute_total_correlation(mdp,
                                N_agents=gridworld.N_agents,
                                agent_state_size_list=agent_state_size_list,
                                agent_action_size_list=agent_action_size_list,
                                f_grad=f_grad,
                                g_grad=g_grad,
                                x=exp_logger_reachability.results[0]['occupancy_vars'])

print('Total corr: {}'.format(total_corr))