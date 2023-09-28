import os
import torch
import torch.nn as nn

from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint_adj

from utils import get_activation

from pdb import set_trace as bp

torch.autograd.set_detect_anomaly(True)

class DataAssimilator(nn.Module):
    def __init__(self, 
                dim_state: int,
                dim_obs: int,
                learn_h: bool = False,
                learn_K: bool = False,
                num_hidden_layers: int = 1,
                layer_width: int = 50,
                activations: int = 'gelu',
                dropout: float = 0.01):
        super(DataAssimilator, self).__init__()

        self.dim_state = dim_state
        self.dim_obs = dim_obs

        self.rhs = HybridODE(dim_state, num_hidden_layers, layer_width, activations, dropout)

        # initialize the observation map to be an unbiased identity map
        self.h_obs = nn.Linear(dim_state, dim_obs, bias=True)
        # set to identity of shape (dim_obs, dim_state)
        self.h_obs.weight.data = torch.eye(dim_obs, dim_state)
        self.h_obs.bias.data = torch.zeros(dim_obs)

        # initialize trainable constant gain K to be transpose of h_obs (K = h_obs^T)
        self.K = nn.Linear(dim_obs, dim_state, bias=False)
        self.K.weight.data = self.h_obs.weight.data.T.clone() # clone to avoid linking these tensors

        # freeze necessary parameters if not learning them
        if not learn_h:
            # freeze the observation map
            for param in self.h_obs.parameters():
                param.requires_grad = False
        
        if not learn_K:
            # freeze the constant gain
            for param in self.K.parameters():
                param.requires_grad = False


        # set an initial condition for the state and register it as a buffer
        # note that this is better than self.x0 = x0 because pytorch-lightning will manage the device
        # so you don't have to do .to(device) every time you use it
        self.register_buffer('x0', torch.zeros(dim_state))

    def solve(self, x0, t):
        # solve the ODE using the initial conditions x0 and time points t
        x = odeint(self.rhs, x0, t, options={'dtype': torch.float32})
        return x

    def forward(self, y_obs, times):
        # y_obs: (N_batch, N_times, dim_obs)
        # times: (N_times)

        # Need to make sure these tensors are on correct device.
        # Easiest way was to assign them to same device as y_obs.
        x_assim = torch.zeros((y_obs.shape[0], y_obs.shape[1], self.dim_state)).to(y_obs).detach()
        x_pred = torch.zeros((y_obs.shape[0], y_obs.shape[1], self.dim_state)).to(y_obs).detach()
        y_pred = torch.zeros_like(y_obs).to(y_obs)
        y_assim = torch.zeros_like(y_obs).to(y_obs)

        x_pred_n = self.x0
        y_pred_n = self.h_obs(self.x0)

        x_pred[:, 0] = x_pred_n.detach()
        y_pred[:, 0] = y_pred_n

        # loop over times
        for n in range(len(times)):

            # perform the filtering/assimilation step w/ constant gain K
            x_assim_n = x_pred_n + self.K( (y_obs[:, n] - y_pred_n) )
            x_assim[:, n] = x_assim_n.detach()
            y_assim[:, n] = self.h_obs(x_assim_n.detach()).detach()

            if n < y_obs.shape[1] - 1:
                # predict the next state by solving the ODE from t_n to t_{n+1} 
                x_pred_n1 = self.solve(x_assim_n, times[n:n+2])[-1]
                x_pred[:, n+1] = x_pred_n1.detach()

                # compute the observation map
                y_pred_n1 = self.h_obs(x_pred_n1)
                y_pred[:, n+1] = y_pred_n1

        return y_pred, y_assim, x_pred, x_assim

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size, 
                 num_hidden_layers=1, 
                 layer_width=50, activations='gelu', dropout=0.01):
        super(FeedForwardNN, self).__init__()

        if not isinstance(layer_width, list):
            layer_width = [layer_width] * (num_hidden_layers)
        
        if not isinstance(activations, list):
            activations = [activations] * (num_hidden_layers)

        # Ensure the number of widths and activations match the number of hidden layers
        assert len(layer_width) == len(activations) == (num_hidden_layers)

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, layer_width[0]))
        layers.append(get_activation(activations[0]))
        layers.append(nn.Dropout(p=dropout))  # Dropout layer added here

        # Hidden layers
        for i in range(1, len(layer_width)):
            layers.append(nn.Linear(layer_width[i-1], layer_width[i]))
            layers.append(get_activation(self.activations[i]))
            layers.append(nn.Dropout(p=dropout))  # Dropout layer added here

        # Output layer
        layers.append(nn.Linear(layer_width[-1], output_size))

        # Sequentially stack the layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class F_Physics(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return 0

# Define a class for the learned ODE model
# This has a forward method to represent a RHS of an ODE, where rhs = f_physics + f_nn
class HybridODE(nn.Module):
    def __init__(self, dim_state, 
                 num_hidden_layers=1, 
                 layer_width=50, activations='gelu', dropout=0.01):
        super(HybridODE, self).__init__()

        self.f_physics = F_Physics() # currently just a placeholder (outputs 0)

        self.f_nn = FeedForwardNN(dim_state, dim_state,
                                  num_hidden_layers, layer_width, activations, dropout)

    def forward(self, t, x, bound=100):

        # if x is outside of [-100,100], set rhs to 0 to achieve fixed point at a boundary instead of blow-up
        if torch.any(torch.abs(x) > bound):
            rhs = torch.zeros_like(x)
        else:
            rhs = self.f_physics(x) + self.f_nn(x)
        return rhs
