import sys
sys.path.append('../../')
import torch
from models.DA.runner import Runner
from utils import dict_combiner
import argparse


# use argparse to get command line argument for which experiment to run
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='mlda_l63_partialNoisyObs_noise')
parser.add_argument('--fast_dev_run', type=bool, default=False)
parser.add_argument('--accelerator', type=str, default='auto')
parser.add_argument('--devices', type=str, default='auto')
parser.add_argument('--run_all', type=bool, default=True)
parser.add_argument('--run_id', type=int, default=0)
args = parser.parse_args()

# build a dict of experimental conditions
exp_dict = {
    'project_name': [args.project_name],
    'fast_dev_run': [args.fast_dev_run],
    'accelerator': [args.accelerator],
    'devices': [args.devices],
    # data settings
    'n_trajectories_train': [10], # smaller dataset for debugging
    'n_trajectories_val': [2],
    'n_trajectories_test': [2],
    'T': [100],
    'T_long': [100], #[1000],
    'train_sample_rate': [1e-2],
    'test_sample_rates': [[0.01]],
    'batch_size': [2**12],
    'batch_length_T': [2], # length of a batch in model time units (e.g., for sample rate 1e-2, this is 200 samples)
    'burnin_frac': [0.75], # fraction of batch used for burn in (loss not computed on predictions made in this portion)
    'dyn_sys_name': ['Lorenz63'],
    'shuffle': ['once'], # options are 'once', 'every_epoch', None
    'normalizer': ['unit_gaussian'], # options are 'inactive', 'unit_gaussian', 'min_max'
    'obs_noise_std': [1, 2, 5, 10],
    # optimizer settings
    'limit_train_batches': [1.0],
    'limit_val_batches': [1.0],
    'limit_test_batches': [1.0],
    'learning_rate': [1e-2],
    'dropout': [0],
    'lr_scheduler_params': [
                            {'patience': 3, 'factor': 0.5},
                             ],
    'max_epochs': [100],
    'monitor_metric': ['loss/val/mse'],
    # model settings
    'dim_obs': [1],
    'dim_state': [3],
    'use_physics': [False],
    'use_nn': [True],
    'learn_h': [False],
    'learn_K': [True],
    'init_K': ['hT'], # options are 'random', 'hT'
    'num_hidden_layers': [1],
    'layer_width': [50],
    'activations': ['gelu'],
    # ODE settings
    'odeint_use_adjoint': [False],
    'odeint_method': ['dopri5'],
    'odeint_rtol': [1e-3],
    'odeint_atol': [1e-3],
    'odeint_options': [{'dtype': torch.float32}],
}

exp_list = dict_combiner(exp_dict)

# Print the length of the experiment list
print('Number of experiments to sweep: ', len(exp_list))

# run the experiment
if args.run_all:
    id_list = list(range(len(exp_list)))
else:
    id_list = [args.run_id]

for i in id_list:
    print('Running experiment ', i)
    Runner(**exp_list[i])



