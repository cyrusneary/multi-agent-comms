import os, time, sys

sys.path.append('..')

from environments.ma_gridworld import MAGridworld
from environments.three_agent_gridworld import ThreeAgentGridworld
from utils.experiment_logger import ExperimentLogger

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
save_file_name = '2022-01-06-12-23-20_ma_gridworld_total_corr_add_end_state_0p05.pkl' # data file used in final submission
# save_file_name = '2022-01-06-14-17-07_three_agent_gridworld_total_corr_0p05.pkl' # Three-agent data file used in final submission
save_str = os.path.join(base_path, save_file_name)

exp_logger = ExperimentLogger(load_file_str=save_str)

##### Create the gridworld from the logged parameters

t_start = time.time()
# gridworld = MAGridworld(**exp_logger.environment_settings)
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
    'walls' : gridworld.walls
}
print('Loaded multiagent gridworld.')
print('Constructed the gridworld in {} seconds.'.format(time.time() - t_start))

##### Create some GIFs

gif_save_folder = os.path.join(os.path.abspath(os.path.curdir), 
                                '..', 'examples', 'gifs')

# # First for the random initial policy
# gridworld.generate_gif(exp_logger.results[-1]['policy'], 
#                         save_folder_str=gif_save_folder,
#                         save_file_name='ma_gridworld_random_init_policy.gif',
#                         use_imaginary_play=False,
#                         num_trajectories=5,
#                         max_steps_per_trajectory=30)

gif_save_name = save_file_name[0:save_file_name.find('.')] + '.gif'

print(exp_logger.results[max(exp_logger.results.keys())])

# Now for the final solution policy
policy = exp_logger.results[max(exp_logger.results.keys())]['policy']
gridworld.generate_gif(policy, 
                        save_folder_str=gif_save_folder,
                        save_file_name=gif_save_name,
                        use_imaginary_play=True,
                        num_trajectories=20,
                        max_steps_per_trajectory=30)