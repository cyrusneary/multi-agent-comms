from cvxpy.error import SolverError
import numpy as np
import datetime
import sys, os, time

sys.path.append('../')

from environments.ma_gridworld import MAGridworld

from optimization_problems.total_corr2 import build_linearized_program
from optimization_problems.joint_entropy import build_joint_entropy_program
from optimization_problems.reachability_LP import build_reachability_LP
from optimization_problems.random_policy_program import build_random_policy_program

from utils.experiment_logger import ExperimentLogger
from utils.process_occupancy import *

curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
exp_name = curr_datetime + '_three_agent_gridworld_total_corr_0p05'
exp_logger = ExperimentLogger(experiment_name=exp_name)

rebuild_gridworld = True
exp_logger.environment_settings = {
    'N_agents' : 3,
    'Nr' : 3,
    'Nc' : 3,
    'slip_p' : 0.05, # 0.05
    'initial_state' : (2, 2, 0, 0, 0, 2),
    'target_states' : [(0, 0, 2, 2, 2, 0)],
    'dead_states' : [],
    'lava' : [],
    'walls' : [],
    'seed' : 1,
}
exp_logger.initial_soln_guess_setup = {
    'type' : 'random', # reachability, entropy
    'settings' : {}
}
exp_logger.optimization_params = {
    'reachability_coef' : 10.0, # 10.0
    'exp_len_coef' : 0.1, # 0.1
    'total_corr_coef' : 1.0 # 4.0
}

### BUILD THE GRIDWOLRD FROM SCRATCH
if rebuild_gridworld:
    # Build the gridworld
    print('Building gridworld')
    t_start = time.time()
    gridworld = MAGridworld(**exp_logger.environment_settings)
    print('Constructed gridworld in {} seconds'.format(time.time() - t_start))

    # Sanity check on the transition matrix
    for s in range(gridworld.Ns_joint):
        for a in range(gridworld.Na_joint):
            assert(np.abs(np.sum(gridworld.T[s, a, :]) - 1.0) <= 1e-12)

    # Save the constructed gridworld
    save_file_str = os.path.join(os.path.abspath(os.path.curdir),
                                    '..', 'environments',
                                    'saved_environments', 'three_agent_gridworld.pkl')
    gridworld.save(save_file_str)

##### LOAD A PRE-CONSTRUCTED GRIDWORLD
else:
    load_file_str = os.path.join(os.path.abspath(os.path.curdir),
                                    '..', 'environments',
                                    'saved_environments', 'three_agent_gridworld.pkl')
    gridworld = MAGridworld(load_file_str=load_file_str)

    # Save the gridworld's settings to the logger.
    exp_logger.environment_settings = {
        'N_agents' : gridworld.N_agents,
        'Nr' : gridworld.Nr,
        'Nc' : gridworld.Nc,
        'slip_p' : gridworld.slip_p,
        'initial_state' : gridworld.initial_state,
        'target_states' : gridworld.target_states,
        'dead_states' : gridworld.dead_states,
        'lava' : gridworld.lava,
        'walls' : gridworld.walls
    }
    print('Loaded multiagent gridworld.')

# Construct the corresponding MDP
mdp = gridworld.build_mdp()

##### Start with an easier optimization problem to get an initial guess.

if exp_logger.initial_soln_guess_setup['type'] == 'random':
    # Construct and solve for a random initial policy
    rand_init = np.random.rand(mdp.Ns, mdp.Na)
    prob_init, vars_init, _ = build_random_policy_program(mdp, 
                                                        rand_init=rand_init)
elif exp_logger.initial_soln_guess_setup['type'] == 'entropy':
    # Construct and solve for an initial policy maximizing entropy
    # exp_len_coef = 0.0, entropy_coef = 0.01
    prob_init, vars_init, _ = build_joint_entropy_program(mdp,
                        **exp_logger.initial_soln_guess_setup['settings'])
elif exp_logger.initial_soln_guess_setup['type'] == 'reachability':
    # Construct and solve the reachability LP
    # exp_len_coef = 0.01
    prob_init, vars_init, params_init = build_reachability_LP(mdp, 
                        **exp_logger.initial_soln_guess_setup['settings'])

##### Build the linearized version of the 
##### total correlation optimization problem

agent_state_size_list = []
agent_action_size_list = []
for agent_id in range(gridworld.N_agents):
    agent_state_size_list.append(gridworld.Ns_local)
    agent_action_size_list.append(gridworld.Na_local)

# Construct and solve the reachability LP
t = time.time()
prob, vars, params, f_grad, g_grad = \
    build_linearized_program(mdp, 
        gridworld.N_agents,
        agent_state_size_list,
        agent_action_size_list,
        gridworld.check_agent_state_action,
        gridworld.check_agent_state,
        reachability_coef=exp_logger.optimization_params['reachability_coef'],
        exp_len_coef=exp_logger.optimization_params['exp_len_coef'],
        total_corr_coef=exp_logger.optimization_params['total_corr_coef'])
print('Constructed optimization problem in {} seconds.'.format(time.time()-t)) 

##### Solve the initial problem to get an initial guess

t = time.time()
prob_init.solve(verbose=True, solver='ECOS')
print('Solved for initial guess in {} seconds'.format(time.time() - t)) 

# Save the initial policy and its statistics.
occupancy_vars_start = process_occupancy_vars(vars_init[0])
policy_start = policy_from_occupancy_vars(mdp, occupancy_vars_start)
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

exp_logger.results[-1] = {
    'occupancy_vars' : occupancy_vars_start,
    'policy' : policy_start,
    'opt_val' : prob_init.solution.opt_val,
    'success_prob' : success_prob,
    'expected_len' : expected_len,
    'joint_entropy' : joint_entropy,
    'total_corr' : total_corr,
}

print(('Success probability: {}, \n \
        expected length: {}, \n \
        joint entropy: {}\n \
        total correlation: {}'.format(success_prob, 
                                    expected_len, 
                                    joint_entropy, 
                                    total_corr)))

x_start = occupancy_vars_start

##### Solve the problem via convex-concave procedure

x_last = x_start
for i in range(100):
    params[3].value = x_last
    prob.solve()

    # Save the results of the current iteration
    occupancy_vars = process_occupancy_vars(vars[0])
    policy = policy_from_occupancy_vars(mdp, occupancy_vars)
    success_prob = success_probability_from_occupancy_vars(mdp, occupancy_vars)
    expected_len = expected_len_from_occupancy_vars(mdp, occupancy_vars)
    joint_entropy = compute_joint_entropy(mdp, occupancy_vars)
    total_corr = compute_total_correlation(mdp,
                                N_agents=gridworld.N_agents,
                                agent_state_size_list=agent_state_size_list,
                                agent_action_size_list=agent_action_size_list,
                                f_grad=f_grad,
                                g_grad=g_grad,
                                x=occupancy_vars)
    empirical_rate = gridworld.empirical_success_rate(policy,
                                                use_imaginary_play=True,
                                                num_trajectories=1000,
                                                max_steps_per_trajectory=200)
    theoretical_bound = success_prob - np.sqrt(1 - np.exp(-total_corr))

    exp_logger.results[i] = {
        'occupancy_vars' : occupancy_vars,
        'policy' : policy,
        'opt_val' : prob.solution.opt_val,
        'x_last' : x_last,
        'success_prob' : success_prob,
        'expected_len' : expected_len,
        'joint_entropy' : joint_entropy,
        'total_corr' : total_corr,
        'empirical_imag_success_rate' : empirical_rate,
        'theoretical_bound' : theoretical_bound,
    }
    print('[{}]: Success probability: {}, \
            total correlation: {},\
            theoretical bound: {}, \
            empirical success rate: {}'.format(i, 
                                        exp_logger.results[i]['success_prob'],
                                        exp_logger.results[i]['total_corr'],
                                        exp_logger.results[i]['theoretical_bound'],
                                        exp_logger.results[i]['empirical_imag_success_rate']))

    x_last = occupancy_vars

    save_folder_str = os.path.join(os.path.abspath(os.path.curdir), 'results')
    save_file_name = exp_logger.experiment_name + '.pkl'
    exp_logger.save(save_folder_str, save_file_name)

# ##### Empirically measure imaginary play success
# t_start = time.time()
# empirical_rate = gridworld.empirical_success_rate(policy,
#                                                 use_imaginary_play=True,
#                                                 num_trajectories=1000,
#                                                 max_steps_per_trajectory=30)
# print('Imaginary play success rate: {} \
#         in {} seconds'.format(empirical_rate, time.time() - t_start))

# ##### Create some GIFs

# gif_save_folder = os.path.join(os.path.abspath(os.path.curdir), 'gifs')

# # First for the random initial policy
# gridworld.generate_gif(policy_start, 
#                         save_folder_str=gif_save_folder,
#                         save_file_name='ma_gridworld_random_init_policy.gif',
#                         use_imaginary_play=False,
#                         num_trajectories=5,
#                         max_steps_per_trajectory=30)

# # Now for the final solution policy
# gridworld.generate_gif(policy, 
#                         save_folder_str=gif_save_folder,
#                         save_file_name='ma_gridworld_total_corr.gif',
#                         use_imaginary_play=False,
#                         num_trajectories=5,
#                         max_steps_per_trajectory=30)