import os, time, sys
import pickle
import numpy as np

from matplotlib.pyplot import grid

sys.path.append('..')

from environments.ma_gridworld import MAGridworld
from environments.three_agent_gridworld import ThreeAgentGridworld
from utils.experiment_logger import ExperimentLogger

# Load the solved policy
base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
# save_file_name = '2022-01-06-12-23-20_ma_gridworld_total_corr_add_end_state_0p05.pkl' # data file used in final submission
save_file_name = '2022-03-30-14-28-33_ma_gridworld_reachability_add_end_state_0p05.pkl' # data file for baseline policy
# save_file_name = '2022-01-06-14-17-07_three_agent_gridworld_total_corr_0p05.pkl' # Three-agent data file used in final submission
save_str = os.path.join(base_path, save_file_name)

exp_logger = ExperimentLogger(load_file_str=save_str)

# Number of trajectories to generate
num_traj = 10
use_imaginary_play = False

##### Create the gridworld from the logged parameters

rebuild_gridworld = True
exp_logger.environment_settings = {
    'N_agents' : 2,
    'Nr' : 5,
    'Nc' : 5,
    'slip_p' : 0.05, # 0.05
    'initial_state' : (4,0,4,4),
    'target_states' : [(4, 3, 4, 1)],
    'dead_states' : [],
    'lava' : [(0, 0), (0,3), (0,1)],
    'walls' : [(0,2), (2,2), (4,2)],
    'seed' : 42,
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
                                    'saved_environments', 'ma_gridworld.pkl')
    gridworld.save(save_file_str)

##### LOAD A PRE-CONSTRUCTED GRIDWORLD
else:
    load_file_str = os.path.join(os.path.abspath(os.path.curdir),
                                    '..', 'environments',
                                    'saved_environments', 'ma_gridworld.pkl')
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
        'walls' : gridworld.walls,
        'seed' : gridworld.seed,
    }
    print('Loaded multiagent gridworld.')

traj_save_folder = os.path.join(os.path.abspath(os.path.curdir), 'trajectory_data')

# Load the final solution policy
policy = exp_logger.results[max(exp_logger.results.keys())]['policy']

# Simulate num_traj trajectories and save them to a list.
trajectory_list = []
for t_ind in range(num_traj):
    if use_imaginary_play:
        traj = gridworld.run_trajectory_imaginary(policy, max_steps=30)

        # Convert from state indexes to position representation
        for i in range(len(traj)):
            traj[i] = gridworld.pos_from_index[traj[i]]
        trajectory_list.append(traj)

        # trajectory_list.append(gridworld.run_trajectory_imaginary(policy, 
        #                     max_steps=30))
    else:
        traj = gridworld.run_trajectory(policy, max_steps=30)

        # Convert from state indexes to position representation
        for i in range(len(traj)):
            traj[i] = gridworld.pos_from_index[traj[i]]
        trajectory_list.append(traj)

# Save the gridworld parameters used to generate the trajectory data
env_data = {
    'Nr' : gridworld.Nr,
    'Nc' : gridworld.Nc,
    'N_agents' : gridworld.N_agents,
    'Ns_local' : gridworld.Ns_local,
    'Na_local' : gridworld.Na_local,
    'Na_joint' : gridworld.Na_joint,
    'initial_state' : gridworld.initial_state,
    'initial_index' : gridworld.initial_index,
    'target_states' : gridworld.target_states,
    'target_indexes' : gridworld.target_indexes,
    'slip_p' : gridworld.slip_p,
    'dead_states' : gridworld.dead_states,
    'dead_indexes' : gridworld.dead_indexes,
    'lava' : gridworld.lava,
    'walls' : gridworld.walls,
    'T' : gridworld.T,
    'seed' : gridworld.seed
}

save_data = {
    'env_data' : env_data,
    'trajectory_list' : trajectory_list
}

save_str = os.path.join(traj_save_folder, save_file_name)

with open(save_str, 'wb') as f:
    pickle.dump(save_data, f)