import os, time, sys

sys.path.append('..')

from environments.ma_gridworld import MAGridworld
from environments.three_agent_gridworld import ThreeAgentGridworld
from utils.experiment_logger import ExperimentLogger
from markov_decision_process.mdp import MDP
from optimization_problems.total_corr_add_end_state import compute_total_correlation, \
                                                    process_occupancy_vars,\
                                                        build_linearized_program

import numpy as np

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
save_file_name = '2021-10-06-15-51-09_ma_gridworld_total_corr_slip_0p05.pkl'
save_str = os.path.join(base_path, save_file_name)

exp_logger = ExperimentLogger(load_file_str=save_str)

##### Create the gridworld from the logged parameters

t_start = time.time()
gridworld = MAGridworld(**exp_logger.environment_settings)
# load_file_str = os.path.join(os.path.abspath(os.path.curdir),
#                                     '..', 'environments',
#                                     'saved_environments', 'ma_gridworld.pkl')

# gridworld = MAGridworld(load_file_str=load_file_str)

# # Save the gridworld's settings to the logger.
# exp_logger.environment_settings = {
#     'N_agents' : gridworld.N_agents,
#     'Nr' : gridworld.Nr,
#     'Nc' : gridworld.Nc,
#     'slip_p' : gridworld.slip_p,
#     'initial_state' : gridworld.initial_state,
#     'target_states' : gridworld.target_states,
#     'dead_states' : gridworld.dead_states,
#     'lava' : gridworld.lava,
#     'walls' : gridworld.walls
# }

print('Loaded multiagent gridworld.')
print('Constructed the gridworld in {} seconds.'.format(time.time() - t_start))

mdp = gridworld.build_mdp()
N_agents = gridworld.N_agents
x_sol_no_aux = exp_logger.results[max(exp_logger.results.keys())]['occupancy_vars']

Na_local = round(pow(mdp.Na, 1/N_agents))
Na = (Na_local + 1) ** N_agents

action_shape = tuple([Na_local + 1 for i in range(N_agents)])

local_auxiliary_action_index = Na_local

true_actions = []
partially_auxiliary_actions = []
all_auxiliary_actions = []
for a_joint in range(Na):
    a_tuple = np.unravel_index(a_joint, action_shape)
    if np.sum(np.array(a_tuple) == local_auxiliary_action_index) == 0:
        true_actions.append(a_joint)
    elif np.sum(np.array(a_tuple) == local_auxiliary_action_index) == N_agents:
        all_auxiliary_actions.append(a_joint)
    else:
        partially_auxiliary_actions.append(a_joint)

x = np.zeros((mdp.Ns, Na))

ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(true_actions)))
ind_cols = np.tile(np.array(true_actions), (len(mdp.active_states), 1))
x[ind_rows, ind_cols] = x_sol_no_aux[mdp.active_states, :]

ind_rows = np.tile(np.array([mdp.active_states]).transpose(), (1, len(true_actions)))
ind_cols = np.tile(np.array(true_actions), (len(mdp.active_states), 1))
occupancy_in = np.hstack(
        [np.sum(np.multiply(x[ind_rows, ind_cols], 
                            mdp.T[mdp.active_states, :, i])) 
            for i in range(mdp.Ns)]
        )

ind_rows = np.tile(np.array([mdp.absorbing_states]).transpose(), (1, len(all_auxiliary_actions)))
ind_cols = np.tile(np.array(all_auxiliary_actions), (len(mdp.absorbing_states), 1))
x[ind_rows, ind_cols] = occupancy_in[mdp.absorbing_states].reshape(x[ind_rows, ind_cols].shape)

success_ind = np.ravel_multi_index((4,4,4,0),(5,5,5,5))

print(x[596,-1])

agent_state_size_list = []
agent_action_size_list = []
for agent_id in range(gridworld.N_agents):
    agent_state_size_list.append(gridworld.Ns_local)
    agent_action_size_list.append(gridworld.Na_local)

prob, vars, params, f_grad, g_grad = \
    build_linearized_program(mdp, 
        gridworld.N_agents,
        agent_state_size_list,
        agent_action_size_list,
        gridworld.check_agent_state_action_with_aux,
        gridworld.check_agent_state,
        reachability_coef=exp_logger.optimization_params['reachability_coef'],
        exp_len_coef=exp_logger.optimization_params['exp_len_coef'],
        total_corr_coef=exp_logger.optimization_params['total_corr_coef'])

x[np.where(x <= 1e-10)] = 1e-10
tc = compute_total_correlation(mdp,
                            N_agents=gridworld.N_agents,
                            agent_state_size_list=agent_state_size_list,
                            agent_action_size_list=agent_action_size_list,
                            f_grad=f_grad,
                            g_grad=g_grad,
                            x=x)

print('tc val: {}'.format(tc))


