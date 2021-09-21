import os, time, sys

sys.path.append('..')

from environments.ma_gridworld import MAGridworld
from utils.experiment_logger import ExperimentLogger

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'examples', 'results'))
save_file_name = '2021-09-20-22-03-37_ma_gridworld_reachability.pkl'
save_str = os.path.join(base_path, save_file_name)

exp_logger = ExperimentLogger(load_file_str=save_str)

##### Create the gridworld from the logged parameters

t_start = time.time()
gridworld = MAGridworld(**exp_logger.environment_settings)
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

# Now for the final solution policy
policy = exp_logger.results[max(exp_logger.results.keys())]['policy']
gridworld.generate_gif(policy, 
                        save_folder_str=gif_save_folder,
                        save_file_name='ma_gridworld_reachability_imag_play.gif',
                        use_imaginary_play=True,
                        num_trajectories=5,
                        max_steps_per_trajectory=30)