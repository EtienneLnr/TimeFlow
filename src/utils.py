import torch
import numpy as np
from torch.utils.data import Dataset
import sys
from pathlib import Path


def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf


def set_seed(seed=33):
    """Set all seeds for the experiments.
    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


class DatasetSamples(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates."""

    def __init__(self, v, grid, latent_dim, sample_ratio_batch=None):
        """
        Args:
            v (torch.Tensor): Dataset values, either x or y
            grid (torch.Tensor): Coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
            with_time (bool, optional): If True, time dimension should be flattened into batch dimension. Defaults to False.
        """
        self.v = v
        self.z = torch.zeros((v.shape[0], latent_dim))
        self.c = grid

        if sample_ratio_batch == None:
            self.n_points = None
        else:
            self.n_points = int(sample_ratio_batch * self.c.shape[1])

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx, full_length=False):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.n_points == None or full_length == True:    
            sample_v = self.v[idx, ...]
            sample_z = self.z[idx, ...]
            sample_c = self.c[idx, ...]
        
        else:
            permutation = torch.randperm(self.c.shape[1])[:self.n_points]
            sample_v = self.v[idx, permutation, ...]
            sample_z = self.z[idx, ...]
            sample_c = self.c[idx, permutation, ...]


        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values



class DatasetSamplesForecasting(Dataset):
    """Custom dataset for forecasting task. Contains the values, the codes, and the coordinates."""

    def __init__(self, 
                v, 
                grid, 
                latent_dim, 
                look_back_window, 
                horizon, 
                length_of_interest, 
                passed_ratio=None,
                horizon_ratio=None):
        """
        Args:
            v (torch.Tensor): Dataset values, either x or y
            grid (torch.Tensor): Coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
            with_time (bool, optional): If True, time dimension should be flattened into batch dimension. Defaults to False.
        """

        self.v = v
        self.c = grid
        self.z = torch.zeros((v.shape[0], latent_dim))
        self.window_size = int(horizon + look_back_window)
        self.length_of_interest = length_of_interest
        self.horizon = horizon
        self.look_back_window = look_back_window

        if passed_ratio == None:
            self.n_points_passed = None
        else:
            self.n_points_passed = int(passed_ratio * look_back_window)

        if horizon_ratio == None:
            self.n_points_horizon = None
        else:
            self.n_points_horizon = int(horizon_ratio * horizon)


    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        end_point = torch.randint(self.window_size, self.length_of_interest, (1,))
        first_point = end_point - self.window_size

        sample_v = self.v[idx, first_point : end_point, :]

        v_passed = sample_v[:self.look_back_window]
        v_horizon = sample_v[-self.horizon:]
        c_passed = self.c[idx, :self.look_back_window]
        c_horizon = self.c[idx, -self.horizon:]

        permutation_passed = torch.randperm(self.look_back_window)[:self.n_points_passed]
        permutation_horizon = torch.randperm(self.horizon)[:self.n_points_horizon]

        real_sample_v_passed = v_passed[permutation_passed, ...]
        real_sample_v_horizon = v_horizon[permutation_horizon, ...]

        sample_z = self.z[idx, :]

        real_sample_c_passed = c_passed[permutation_passed, ...]
        real_sample_c_horizon = c_horizon[permutation_horizon, ...]

        return real_sample_v_passed, real_sample_v_horizon, sample_z, real_sample_c_passed, real_sample_c_horizon, idx

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values




def fixed_subsampling_series_imputations(dataset_name, 
                                         draw_ratio, 
                                         version=0,
                                         setting='classic',
                                         train_or_test='train'):

    '''
    Description
        Draw subsampling for specific version

    Arguments
        @dataset_name : in {'Solar', 'Electricity', 'Traffic'}
        @draw ratio : in {0.05, 0.1, 0.2, 0.3, 0.5}
        @version in {0, 1, 2, 3, 4}
        @setting : classic, train_test

    Return
        @series
        @grid
    '''
    DIR = str(Path(__file__).parents[1]) + '/data/' + dataset_name + '/Imputation/'

    if setting == 'classic':

        RESULTS_DIR = DIR + 'Classic_framework/X_subsampling_version_' + str(version) 
        FILE_NAME =  RESULTS_DIR + '/small_X_' + str(draw_ratio) + '.pt'
        dico = torch.load(FILE_NAME, map_location=torch.device('cpu'))

        grid = dico['small_grid']
        series = dico['small_data']
        permutations = dico['permutations']


    elif setting == 'train_test':

        if train_or_test == 'train' or train_or_test == 'test':

            RESULTS_DIR = DIR + 'Train_test_framework/X_subsampling_version_' + str(version) 
            FILE_NAME =  RESULTS_DIR + '/small_X_' + train_or_test + '_' + str(draw_ratio) + '.pt'
            dico = torch.load(FILE_NAME, map_location=torch.device('cpu'))

            grid = dico['small_grid']
            series = dico['small_data']
            permutations = dico['permutations']

        else:
            'not supported value'

    else:
        print("not supported setting")

    return series, grid, permutations




def fixed_sampling_series_imputations(dataset_name,  
                                      version=0,
                                      setting='classic',
                                      train_or_test='train'):
    
    """
    Description
        Draw whole sequence for specific version
    """

    DIR = str(Path(__file__).parents[1]) + '/data/' + dataset_name + '/Imputation/'

    if setting == 'classic':

        RESULTS_DIR = DIR + 'Classic_framework/X_subsampling_version_' + str(version) 
        FILE_NAME =  RESULTS_DIR + '/X_complete.pt'
        dico = torch.load(FILE_NAME, map_location=torch.device('cpu'))
        grid = dico['grid']
        series = dico['data']


    elif setting == 'train_test':

        if train_or_test == 'train' or train_or_test == 'test':

            RESULTS_DIR = DIR + 'Train_test_framework/X_subsampling_version_' + str(version) 
            FILE_NAME =  RESULTS_DIR + '/X_' + train_or_test + '_complete.pt'
            dico = torch.load(FILE_NAME, map_location=torch.device('cpu'))
            grid = dico['grid']
            series = dico['data']

        else:
            'not supported value'

    else:
        print("not supported setting")


    return series, grid



def fixed_sampling_series_forecasting(dataset_name, 
                                      horizon, 
                                      version=0,
                                      setting='classic',
                                      train_or_test='train'):

    '''
    Description
        Draw forecast sample

    Arguments
        @dataset_name : in {'Electricity', 'Traffic'}
        @horizon : in {96, 192, 336, 720}
        @version in {0, 1, 2, 3, 4}
        @setting : classic, train_test

    Return
        @series
        @grid
    '''

    DIR = str(Path(__file__).parents[1]) + '/data/' + dataset_name + '/Forecasting/'

    if setting == 'classic':

        RESULTS_DIR = DIR + 'Classic_framework/X_forecasting_version_' + str(version) 
        FILE_NAME_PASSED =  RESULTS_DIR + '/X_passed_horizon_' + str(horizon) + '.pt'
        FILE_NAME_TARGET =  RESULTS_DIR + '/X_target_horizon_' + str(horizon) + '.pt'
        dico_passed = torch.load(FILE_NAME_PASSED, map_location=torch.device('cpu'))
        dico_target = torch.load(FILE_NAME_TARGET, map_location=torch.device('cpu'))

        grid = dico_passed['grid']
        series_passed = dico_passed['X_passed']
        series_target = dico_target['X_target']

    elif setting == 'train_test':

        if train_or_test == 'train' or train_or_test == 'test':

            RESULTS_DIR = DIR + 'Train_test_framework/X_forecasting_version_' + str(version)
            FILE_NAME_PASSED =  RESULTS_DIR + '/X_passed_horizon_' + train_or_test + '_' + str(horizon) + '.pt'
            FILE_NAME_TARGET =  RESULTS_DIR + '/X_target_horizon_' + train_or_test +  '_' + str(horizon) + '.pt'
            dico_passed = torch.load(FILE_NAME_PASSED, map_location=torch.device('cpu'))
            dico_target = torch.load(FILE_NAME_TARGET, map_location=torch.device('cpu'))

            grid = dico_passed['grid']
            series_passed = dico_passed['X_passed']
            series_target = dico_target['X_target']

        else:
            'not supported value'

    else:
        print("not supported setting")

    return series_passed, series_target, grid


def z_normalize(X):

    X_mean = X[:,:,0].mean(dim=1)
    X_std = X[:,:,0].std(dim=1)
    X_normalize = (X[:,:,0].transpose(1,0) - X_mean) / (X_std + 1e-7)

    return X_normalize.transpose(1,0).unsqueeze(-1)


def z_normalize_out(X):
    
    X_mean = X[:,:,0].mean(dim=1)
    X_std = X[:,:,0].std(dim=1)
    X_normalize = (X[:,:,0].transpose(1,0) - X_mean) / (X_std + 1e-7)

    return X_normalize.transpose(1,0).unsqueeze(-1), X_mean, X_std



def z_denormalize_out(X_norm, X_mean, X_std):

    X = X_norm[:,:,0].transpose(1,0) * (X_std + 1e-7) + X_mean 

    return X.transpose(1,0).unsqueeze(-1)



def z_normalize_other(X, X_mean, X_std):

    X_normalize = (X[:,:,0].transpose(1,0) - X_mean) / (X_std + 1e-7)

    return X_normalize.transpose(1,0).unsqueeze(-1)



def convert_error(sample_r, score_learned, total_score):
    partial_score = (total_score - sample_r * score_learned) * 1 / (1-sample_r)
    return partial_score