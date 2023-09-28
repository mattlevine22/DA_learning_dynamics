import torch
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import pytorch_lightning as pl
from sklearn.utils import shuffle as skshuffle
import random
from utils import InactiveNormalizer, UnitGaussianNormalizer, MaxMinNormalizer

from pdb import set_trace as bp

def load_dyn_sys_class(dataset_name):
    dataset_classes = {
        'Lorenz63': Lorenz63,
        'Rossler': Rossler,
        # Add more dataset classes here for other systems
    }

    if dataset_name in dataset_classes:
        return dataset_classes[dataset_name]
    else:
        raise ValueError(f"Dataset class '{dataset_name}' not found.")

class DynSys(object):
    def __init__(self, state_dim=1, obs_noise_std=1):
        self.state_dim = state_dim
        self.obs_noise_std = obs_noise_std
    
    def rhs(self, t, x):
        raise NotImplementedError
    
    def get_inits(self, size):
        raise NotImplementedError
    
    def solve(self, N_traj, T, dt):
        '''ode solver for the dynamical system.
        Returns xyz, times, where:
        xyz is a tensor of shape (N_traj, N_times, state_dim)
        times is a tensor of shape (N_times, 1)
        '''
        times = torch.arange(0, T, dt)
        xyz0 = self.get_inits(N_traj)
        xyz = odeint(self.rhs, xyz0, times)
        return xyz.permute(1, 0, 2), times.reshape(-1, 1)

    def h_obs(self, x):
        '''observation function: default is to observe the first component of the state'''
        return x[..., 0:1]
    
    def noisy_obs(self, x):
        '''default is to apply additive i.i.d. zero-mean Gaussian noise to observations'''
        y = self.h_obs(x)
        y_noisy = y + self.obs_noise_std * torch.randn_like(y)
        return y_noisy
    
class Lorenz63(DynSys):
    def __init__(self, state_dim=3, sigma=10, rho=28, beta=8/3):
        super().__init__(state_dim=state_dim)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def rhs(self, t, x):
        x, y, z = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return torch.cat([dx, dy, dz], dim=1)
    
    def get_inits(self, size):
        x0 = torch.empty(size, 1).uniform_(-15, 15)
        y0 = torch.empty(size, 1).uniform_(-15, 15)
        z0 = torch.empty(size, 1).uniform_(0, 40)
        xyz0 = torch.cat([x0, y0, z0], dim=1)
        return xyz0

class Rossler(DynSys):
    def __init__(self, state_dim=2, a=0.2, b=0.2, c=5.7):
        super().__init__(state_dim=state_dim)
        self.a = a
        self.b = b
        self.c = c
    
    def rhs(self, t, x):
        x, y, z = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return torch.cat([dx, dy, dz], dim=1)
    
    def get_inits(self, size):
        x0 = torch.empty(size, 1).uniform_(-10, 15)
        y0 = torch.empty(size, 1).uniform_(-30, 0)
        z0 = torch.empty(size, 1).uniform_(0, 30)
        xz0 = torch.cat([x0, y0, z0], dim=1)
        return xz0

class DynamicsDataset(Dataset):
    def __init__(self, 
                 N_traj=1000, 
                 T=1, 
                 sample_rate=0.01, 
                 batch_length=1000,
                 params={},
                 dyn_sys_name='Lorenz63',
                 **kwargs):
        '''use params to pass in parameters for the dynamical system'''
        self.N_traj = N_traj
        self.T = T
        self.sample_rate = sample_rate
        self.batch_length = batch_length 
        # batch idx will return a subset of a trajectory of length batch_length
        self.dynsys = load_dyn_sys_class(dyn_sys_name)(**params)

        self.generate_data()

        # number of sequences of length batch_length in each trajectory
        self.n_batches_per_traj = self.y_obs.shape[1] - self.batch_length + 1


    def generate_data(self):
        # Seq_len, Size (N_traj), state_dim
        xyz, times = self.dynsys.solve(N_traj=self.N_traj, T=self.T, dt=self.sample_rate)
            
        # self.times is (N_times, 1), since all trajectories currently share timepoints
        self.times = times

        # self.x_true is (N_traj, N_times, dim_state)
        self.x_true = xyz

        # self.y_true should be: (N_traj, N_times, dim_obs) and is NOT noisy!
        self.y_true = self.dynsys.h_obs(self.x_true)

        # self.y_obs should be: (N_traj, N_times, dim_obs) and is noisy!
        self.y_obs = self.dynsys.noisy_obs(self.x_true)

        #normalize data
        self.y_obs_normalizer = UnitGaussianNormalizer(
            self.y_obs.reshape(-1, self.y_obs.shape[-1]).data.numpy())

        self.y_obs = self.y_obs_normalizer.encode(self.y_obs)
        self.y_true = self.y_obs_normalizer.encode(self.y_true)


    def __len__(self):
        # the number of batches is equal to the number of trajectories * the number of subsets (of length batch_length) of each trajectory
        return self.N_traj * self.n_batches_per_traj

    def __getitem__(self, idx):
        '''return a batch of data where idx is the batch index.
        The batch idx defines 1 trajectory and a specific subset of that trajectory of length batch_length.
        Across all batches, we will cover the entirety of each trajectory.
        '''

        ## compute batch indices
        # get the trajectory index
        traj_idx = idx // self.n_batches_per_traj
        # get the start index of the batch
        start_idx = idx % self.n_batches_per_traj

        ## collect the batches
        # observations
        y_obs_batch = self.y_obs[traj_idx, start_idx:start_idx+self.batch_length]

        # true states
        x_true_batch = self.x_true[traj_idx, start_idx:start_idx+self.batch_length]

        # true observations
        y_true_batch = self.y_true[traj_idx, start_idx:start_idx+self.batch_length]

        # times
        times_batch = self.times[start_idx:start_idx+self.batch_length]

        return y_obs_batch, x_true_batch, y_true_batch, times_batch


class DynamicsDataModule(pl.LightningDataModule):
    def __init__(self,
            shuffle='once', # can be 'once', 'every_epoch', or 'never'
            batch_size=64,
            batch_length=1000,
            N_traj={'train': 10, 'val': 2, 'test': 2},
            T={'train': 100, 'val': 100, 'test': 100},
            train_sample_rate=0.01,
            test_sample_rates=[0.01],
            params={},
            dyn_sys_name='Lorenz63',
            **kwargs
            ):
        super().__init__()
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.N_traj = N_traj
        self.T = T
        self.train_sample_rate = train_sample_rate
        self.test_sample_rates = test_sample_rates
        self.params = params
        self.dyn_sys_name = dyn_sys_name
        self.shuffle = shuffle

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.train = DynamicsDataset(N_traj=self.N_traj['train'],
                                        T=self.T['train'],
                                        sample_rate=self.train_sample_rate,
                                        params=self.params,
                                        batch_length=self.batch_length,
                                        dyn_sys_name=self.dyn_sys_name)

        self.val = DynamicsDataset(N_traj=self.N_traj['val'],
                                        T=self.T['val'],
                                        sample_rate=self.train_sample_rate,
                                        params=self.params,
                                        batch_length=self.batch_length,
                                        dyn_sys_name=self.dyn_sys_name)

        # build a dictionary of test datasets with different sample rates
        self.test = {}
        for dt in self.test_sample_rates:
            self.test[dt] = DynamicsDataset(N_traj=self.N_traj['test'],
                                        T=self.T['test'],
                                        sample_rate=dt,
                                        params=self.params,
                                        batch_length=self.batch_length,
                                        dyn_sys_name=self.dyn_sys_name)

    def get_dataloader(self, data):
        if self.shuffle == 'once':
            shuffle = False
            data = skshuffle(data)
        elif self.shuffle == 'every_epoch':
            shuffle = True
        else:
            shuffle = False
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        return self.get_dataloader(self.train)

    def val_dataloader(self):
        return self.get_dataloader(self.val)

    def test_dataloader(self, sample_rate=None):
        return {dt: self.get_dataloader(self.test[dt]) for dt in self.test_sample_rates}
