import sys
sys.path.append('../../')
from models.DA.runner import Runner
from utils import dict_combiner
import argparse

# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='mlda_l63_partialNoisyObs')
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--fast_dev_run', type=bool, default=False)
args = parser.parse_args()

# build a dict of experimental conditions
exp_dict = {
    'project_name': [args.project_name],
    'fast_dev_run': [args.fast_dev_run],
    # data settings
    'n_trajectories_train': [10], # smaller dataset for debugging
    'n_trajectories_val': [2],
    'n_trajectories_test': [2],
    'T': [100],
    'train_sample_rate': [0.01],
    'test_sample_rates': [[0.01]],
    'batch_size': [8],
    'batch_length': [200],
    'n_burnin': [190],
    'dyn_sys_name': ['Lorenz63'],
    'shuffle': ['once'], # options are 'once', 'every_epoch', None
    'normalizer': ['inactive'], # options are 'inactive', 'unit_gaussian', 'min_max'
    # optimizer settings
    'limit_train_batches': [0.002],
    'limit_val_batches': [0.005],
    'limit_test_batches': [0.005],
    'learning_rate': [1e-1],
    'dropout': [0],
    'lr_scheduler_params': [
                            {'patience': 3, 'factor': 0.5},
                             ],
    'max_epochs': [100],
    'monitor_metric': ['loss/val/mse'],
    # model settings
    'dim_obs': [1],
    'dim_state': [3],
    'use_physics': [True],
    'use_nn': [False],
    'learn_h': [False],
    'learn_K': [True],
    'num_hidden_layers': [1],
    'layer_width': [50],
    'activations': ['gelu'],
}

exp_list = dict_combiner(exp_dict)

# Print the length of the experiment list
print('Number of experiments to sweep: ', len(exp_list))

# run the experiment
Runner(**exp_list[args.id])



