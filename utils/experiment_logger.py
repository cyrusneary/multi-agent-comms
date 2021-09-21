import pickle
import os

from numpy import save

class ExperimentLogger(object):
    """
    Class to log experiments and parameter sweeps.
    """

    def __init__(self,
                experiment_name : str = '',
                load_file_str : str = ''):
        """
        Constructor for the results logger object.

        Parameters
        ----------
        experiment_name :
            A name for the experiment. 
        load_file_str :
            String representing the path of a data file to use to load a
            saved experiment logger objects. If load_file_str = '', 
            a new experiment logger is created instead.
        """
        if load_file_str == '':
            self.experiment_name = experiment_name
            self.results = {}
            self.environment_settings = {}
            
            # default the any initial solution guesses to be random
            self.initial_soln_guess_setup = {
                'type' : 'random',
                'settings' : {}
            }

            # Can either contain an individual set of parameters or a
            # a collection of parameters.
            self.optimization_params = {}
        else:
            self.load(load_file_str)

    def save(self,
            save_folder_str : str,
            save_file_name : str):
        """
        Save the experiment logger to a file.
        """
        self.save_folder_str = save_folder_str
        self.save_file_name = save_file_name

        save_dict = {}
        save_dict['experiment_name'] = self.experiment_name
        save_dict['results'] = self.results
        save_dict['save_folder_str'] = self.save_folder_str
        save_dict['save_file_name'] = self.save_file_name
        save_dict['environment_settings'] = self.environment_settings
        save_dict['initial_soln_guess_setup'] = self.initial_soln_guess_setup
        save_dict['optimization_params'] = self.optimization_params

        save_file_str = os.path.abspath(os.path.join(save_folder_str, 
                                                        save_file_name))   
        with open(save_file_str, 'wb') as f:
            pickle.dump(save_dict, f) 

    def load(self,
            load_file_str : str):
        """
        Load the experiment logger object from a file.
        """
        with open(load_file_str, 'rb') as f:
            save_dict = pickle.load(f)

        self.experiment_name = save_dict['experiment_name']
        self.results = save_dict['results']
        self.save_folder_str = save_dict['save_folder_str']
        self.save_file_name = save_dict['save_file_name']
        self.environment_settings = save_dict['environment_settings']
        self.initial_soln_guess_setup = save_dict['initial_soln_guess_setup']
        self.optimization_params = save_dict['optimization_params']

