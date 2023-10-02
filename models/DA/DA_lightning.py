import os
import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from models.DA.DA_pytorch import DataAssimilator

from pdb import set_trace as bp

# Define the pytorch lightning module for training the Simple Encoder model
class DataAssimilatorModule(pl.LightningModule):
    def __init__(self, 
                 dim_obs=1,
                 dim_state=10,
                 ode=None,
                 use_physics=False,
                 use_nn=True,
                 num_hidden_layers=1,
                 learn_h=False,
                 learn_K=False,
                 layer_width=50,
                 n_burnin=200,
                 learning_rate=0.01, 
                 activation='gelu',
                 monitor_metric='train_loss',
                 lr_scheduler_params={'patience': 3,
                                      'factor': 0.5},
                 dropout=0.1,
                 T_long=1000,
                 dt_long=0.01,
                 **kwargs):
        super(DataAssimilatorModule, self).__init__()

        self.dim_obs = dim_obs
        self.dim_state = dim_state
        self.n_burnin = n_burnin # number of burn-in steps to ignore when computing loss

        self.first_forward = True # for plotting model-related things once at beginnning of training
        self.learning_rate = learning_rate
        self.monitor_metric = monitor_metric
        self.lr_scheduler_params = lr_scheduler_params

        # initial condition for the long trajectory
        self.x0_inv = torch.zeros(1, dim_state) + 0.1

        # time points for the long trajectory
        self.t_inv = torch.arange(0, T_long, dt_long)

        # initialize the model
        self.model = DataAssimilator(dim_state=dim_state, 
                                     dim_obs=dim_obs,
                                     ode=ode,
                                     use_physics=use_physics,
                                     use_nn=use_nn,
                                     num_hidden_layers=num_hidden_layers,
                                     layer_width=layer_width,
                                     dropout=dropout,
                                     activations=activation,
                                     learn_h=learn_h,
                                     learn_K=learn_K)

    def long_solve(self, device='cpu'):
        '''This function solves the ODE for a long time, and returns the entire trajectory'''
        # solve the ODE using the initial conditions x0 and time points t
        x = self.model.solve(self.x0_inv.to(device), self.t_inv.to(device))
        # x is (N_times, N_batch, dim_state)
        return x

    def forward(self, y_obs, times):
        # since times currently have the same SPACING across all batches, we can reduce this to just the first batch
        times = times[0].squeeze()
        y_pred, y_assim, x_pred, x_assim = self.model(y_obs=y_obs, times=times)
        return y_pred, y_assim, x_pred, x_assim

    def training_step(self, batch, batch_idx):
        y_obs, x_true, y_true, times = batch
        y_pred, y_assim, x_pred, x_assim = self.forward(y_obs, times)
        loss = F.mse_loss(y_pred[:, self.n_burnin:], y_obs[:, self.n_burnin:])
        self.log("loss/train/mse", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_pred[:, self.n_burnin:] - y_obs[:, self.n_burnin:]))
        self.log("loss/train/sup", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self.make_batch_figs(y_obs, x_true, y_true, times, y_pred, x_pred, x_assim, y_assim, tag='Train')

        return loss

    def on_after_backward(self):
        self.log_gradient_norms(tag='afterBackward')

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer and its gradient
        # If using mixed precision, the gradients are already unscaled here
        self.log_gradient_norms(tag='beforeOptimizer')
        self.log_parameter_norms(tag='beforeOptimizer')

        self.log_matrix(self.model.K.weight.detach(), tag='K')
        self.log_matrix(self.model.h_obs.weight.detach(), tag='h_obs')
    
    def log_matrix(self, matrix, tag=''):
        # log the learned constant gain K self.model.K.weight.detach()
        param_dict = {}
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                param_dict[f'parameters/{tag}_{i}{j}'] = matrix[i][j]
        wandb.log(param_dict)

    def log_gradient_norms(self, tag=''):
        norm_type = 2.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().norm(norm_type)
                name = name.replace('.', '_')
                self.log(f"grad_norm/{tag}/{name}", grad_norm,
                         on_step=False, on_epoch=True, prog_bar=False)

    def log_parameter_norms(self, tag=''):
        norm_type = 2.0
        for name, param in self.named_parameters():
            param_norm = param.detach().norm(norm_type)
            name = name.replace('.', '_')
            self.log(f"param_norm/{tag}/{name}", param_norm,
                     on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        y_obs, x_true, y_true, times = batch
        y_pred, y_assim, x_pred, x_assim = self.forward(y_obs, times)
        loss = F.mse_loss(y_pred[:, self.n_burnin:], y_obs[:, self.n_burnin:])
        self.log("loss/val/mse", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_pred[:, self.n_burnin:] - y_obs[:, self.n_burnin:]))
        self.log("loss/val/sup", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            # run the model on the long trajectory
            x_long = self.long_solve(device=y_obs.device)
            y_long = self.model.h_obs(x_long).detach().cpu().numpy()
            
            self.make_batch_figs(y_obs, x_true, y_true, times, y_pred, x_pred, x_assim, y_assim, y_long=y_long, tag='Val')
        return loss

    def make_batch_figs(self, y_obs, x_true, y_true, times, y_pred, x_pred, x_assim, y_assim, y_long=None, tag='', n_examples=2):
        '''This function makes plots for a single batch of data.'''
        if n_examples > x_true.shape[0]:
            n_examples = x_true.shape[0]

        for idx in range(n_examples):
            y_obs_idx = y_obs[idx].detach().cpu().numpy()
            y_true_idx = y_true[idx].detach().cpu().numpy()
            x_true_idx = x_true[idx].detach().cpu().numpy()
            times_idx = times[idx].detach().cpu().numpy()
            y_pred_idx = y_pred[idx].detach().cpu().numpy()
            x_pred_idx = x_pred[idx].detach().cpu().numpy()
            x_assim_idx = x_assim[idx].detach().cpu().numpy()
            y_assim_idx = y_assim[idx].detach().cpu().numpy()


            # Plot Trajectories
            n_rows = y_obs.shape[-1] + 2
            plt.figure()
            fig, axs = plt.subplots(
                nrows=n_rows, ncols=3, figsize=(30, 6 * n_rows), 
                gridspec_kw={'width_ratios': [2.5, 1, 1]},
                sharex='col', squeeze=False)
            
            # master title
            fig.suptitle(f'{tag} Trajectories for Index {idx} w/ Predicted Invariant Measure')
            for i in range(n_rows - 2):
                ax = axs[i, 0]
                # plot the assimilated state of the i'th observation
                ax.plot(times_idx, y_assim_idx[:, i], 
                           ls='',
                           marker='x', 
                           markersize=5,
                            color='blue', label='Assimilated')

                # plot the predicted state of the i'th observation
                ax.plot(times_idx, y_pred_idx[:, i], 
                        ls='',
                        marker='o',
                        markersize=5,
                        markerfacecolor='none',
                        color='blue', label='Prediction')
                
                # plot the noisy observations that we are fitting to
                ax.plot(times_idx, y_obs_idx[:, i], 
                           ls='',
                           marker='o',
                           markersize=5,
                           alpha=0.5,
                           color='red', label='Observation')

                # plot true state of the i'th observation
                ax.plot(times_idx, y_true_idx[:, i], linewidth=3,
                        color='black', label='Ground Truth')

                ax.set_xlabel('Time')
                ax.set_ylabel(f'Observation {i}')
                ax.set_title(
                    f'Observation for component {i} (Index {idx})')
                if i == 0:
                    ax.legend()
                
                # in the second column, plot the same as the first column, but only for the last 20 time steps
                ax = axs[i, 1]
                ax.plot(times_idx[-20:], y_assim_idx[-20:, i],
                        ls='',
                        marker='x', 
                        markersize=10,
                        color='blue', label='Assimilated')
                ax.plot(times_idx[-20:], y_pred_idx[-20:, i],
                        ls='',
                        marker='o',
                        markersize=10,
                        markerfacecolor='none',
                        color='blue', label='Prediction')
                ax.plot(times_idx[-20:], y_obs_idx[-20:, i],
                        ls='',
                        marker='o',
                        markersize=10,
                        alpha=0.5,
                        color='red', label='Observation')
                ax.plot(times_idx[-20:], y_true_idx[-20:, i], linewidth=3,
                        color='black', label='Ground Truth')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'Observation {i}')
                ax.set_title(
                    f'Observation for component {i} (Index {idx})')
                if i == 0:
                    ax.legend()


                # in the third column, plot a kernel density estimate for the distribution of y_long
                if y_long is not None:
                    ax = axs[i, 2]
                    # use seaborn kdeplot
                    sns.kdeplot(y_long[..., i].squeeze(), ax=ax, fill=True, color='blue')
                    ax.set_xlabel(f'Observation {i}')
                    ax.set_ylabel('Density')
                    ax.set_title(f'Invariant Distribution for Observation {i}')

            # plot the learned latent variables
            ax = axs[-2, 0]
            ax.plot(times_idx, x_assim_idx, ls='', marker='x', 
                    markersize=10, 
                    color='gray', label='Assimilated')
            ax.plot(times_idx, x_pred_idx, ls='', marker='o',
                    markersize=10,
                    markerfacecolor='none',
                    color='gray', label='Prediction')
            ax.set_title(f'Learned Latent Variables')
            
            ax = axs[-2, 1]
            ax.plot(times_idx[-20:], x_assim_idx[-20:], ls='', marker='x',
                    markersize=10,
                    color='gray', label='Assimilated')
            ax.plot(times_idx[-20:], x_pred_idx[-20:], ls='', marker='o',
                    markersize=10,
                    markerfacecolor='none',
                    color='gray', label='Prediction')
            ax.set_title(f'Learned Latent Variables')

            # plot the true latent variables
            ax = axs[-1, 0]
            ax.plot(times_idx, x_true_idx, linewidth=3,
                    color='gray', label='Ground Truth')
            ax.set_title(f'True Latent Variables')
            ax.set_xlabel('Time')

            ax = axs[-1, 1]
            ax.plot(times_idx[-20:], x_true_idx[-20:], linewidth=3,
                    color='gray', label='Ground Truth')
            ax.set_title(f'True Latent Variables')
            ax.set_xlabel('Time')
            
            plt.subplots_adjust(hspace=0.5)
            wandb.log({f"plots/{tag}/Trajectories_{idx}": wandb.Image(fig)})
            plt.close('all')


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        dt = self.trainer.datamodule.test_sample_rates[dataloader_idx]
        y_obs, x_true, y_true, times = batch

        y_pred, y_assim, x_pred, x_assim = self.forward(y_obs, times)
        loss = F.mse_loss(y_pred[:, self.n_burnin:], y_obs[:, self.n_burnin:])
        self.log(f"loss/test/mse/dt{dt}", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_pred[:, self.n_burnin:] - y_obs[:, self.n_burnin:]))
        self.log(f"loss/test/sup/dt{dt}", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)

        # log plots
        if batch_idx == 0:
            self.make_batch_figs(y_obs, x_true, y_true, times, y_pred, x_pred, x_assim, y_assim, tag=f'Test/dt{dt}')

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        config = {
            # REQUIRED: The scheduler instance
            "scheduler": ReduceLROnPlateau(optimizer, verbose=True, **self.lr_scheduler_params),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor_metric,  # "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": config,
        }
