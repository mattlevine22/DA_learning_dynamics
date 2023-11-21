import os
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from utils import get_activation, odeint_wrapper, Symmetric, MatrixExponential

from pdb import set_trace as bp

torch.autograd.set_detect_anomaly(True)


class DataAssimilator(nn.Module):
    def __init__(
        self,
        dim_state: int,
        dim_obs: int,
        learn_h: bool = False,
        learn_ObsCov: bool = False,
        learn_StateCov: bool = False,
        ode: object = None,
        odeint_params: dict = {
            "use_adjoint": False,
            "method": "dopri5",
            "rtol": 1e-7,
            "atol": 1e-9,
            "options": {"dtype": torch.float32},
        },
        use_physics: bool = False,
        use_nn: bool = True,
        num_hidden_layers: int = 1,
        layer_width: int = 50,
        activations: int = "gelu",
        dropout: float = 0.01,
    ):
        super(DataAssimilator, self).__init__()

        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.odeint_params = odeint_params

        self.rhs = HybridODE(
            dim_state,
            ode,
            use_physics,
            use_nn,
            num_hidden_layers,
            layer_width,
            activations,
            dropout,
        )

        # initialize the observation map to be an unbiased identity map
        self.h_obs = nn.Linear(dim_state, dim_obs, bias=True)
        # set to identity of shape (dim_obs, dim_state)
        self.h_obs.weight.data = torch.eye(dim_obs, dim_state)
        self.h_obs.bias.data = torch.zeros(dim_obs)
        if not learn_h:
            print("Not learning h")
            # freeze the observation map
            for param in self.h_obs.parameters():
                param.requires_grad = False


        self.Gamma_cov = nn.Linear(dim_obs, dim_obs, bias=False)
        if learn_ObsCov:
            self.Gamma_cov.weight.data = torch.zeros(dim_obs) + 0.01 * torch.randn(
                dim_obs, dim_obs
            )
            # torch.abs(self.Gamma_cov.weight.data)
            parametrize.register_parametrization(self.Gamma_cov, "weight", Symmetric())
            parametrize.register_parametrization(
                self.Gamma_cov, "weight", MatrixExponential()
            )
        else:
            print("Not learning Gamma")
            self.Gamma_cov.weight.data = torch.eye(dim_obs, dim_obs)
            # freeze the observation noise covariance
            for param in self.Gamma_cov.parameters():
                param.requires_grad = False

        print("Initial Gamma_cov: ", self.Gamma_cov.weight.data)

        # initialize the state noise covariance to be 0.1*identity
        self.C_cov = nn.Linear(dim_state, dim_state, bias=False)
        if learn_StateCov:
            self.C_cov.weight.data = torch.zeros(dim_state) + 0.01 * torch.randn(
                dim_state, dim_state
            )
            parametrize.register_parametrization(self.C_cov, "weight", Symmetric())
            parametrize.register_parametrization(self.C_cov, "weight", MatrixExponential())
        else:
            print("Not learning C")
            self.C_cov.weight.data = torch.eye(dim_state, dim_state)
            # freeze the state noise covariance
            for param in self.C_cov.parameters():
                param.requires_grad = False

        print("Initial C_cov: ", self.C_cov.weight.data)

        # create scale parameters in SDTDEV units to hopefully make learning easier
        self.Gamma_scale = nn.Parameter(
            torch.tensor(1.0)
        )  # this is the scale of the observation noise STDEV
        self.C_scale = nn.Parameter(
            torch.tensor(1.0)
        )  # this is the scale of the state noise STDEV

        self.compute_K()
        print("Initial K: ", self.K)

        # set an initial condition for the state and register it as a buffer
        # note that this is better than self.x0 = x0 because pytorch-lightning will manage the device
        # so you don't have to do .to(device) every time you use it
        self.register_buffer("x0", torch.zeros(dim_state, requires_grad=True))

    def solve(self, x0, t, params={}):
        # solve the ODE using the initial conditions x0 and time points t
        x = odeint_wrapper(self.rhs, x0, t, **params)
        return x

    def compute_K(self):
        H = self.h_obs.weight
        Gamma_cov = self.Gamma_scale**2 * self.Gamma_cov.weight
        C_cov = self.C_scale**2 * self.C_cov.weight
        self.cov = H @ C_cov @ H.T + Gamma_cov
        self.K = C_cov @ H.T @ torch.inverse(self.cov)

    def forward(self, y_obs, times):
        # y_obs: (N_batch, N_times, dim_obs)
        # times: (N_times)

        # update self.K and self.cov
        self.compute_K()

        # Need to make sure these tensors are on correct device.
        # Easiest way was to assign them to same device as y_obs.
        x_assim = (
            torch.zeros((y_obs.shape[0], y_obs.shape[1], self.dim_state))
            .to(y_obs)
            .detach()
        )
        x_pred = (
            torch.zeros((y_obs.shape[0], y_obs.shape[1], self.dim_state))
            .to(y_obs)
            .detach()
        )
        y_pred = torch.zeros_like(y_obs).to(y_obs)
        y_assim = torch.zeros_like(y_obs).to(y_obs).detach()

        x_pred_n = torch.zeros((y_obs.shape[0], self.dim_state)).to(y_obs)
        x_pred_n[:] = self.x0
        y_pred_n = self.h_obs(x_pred_n)

        x_pred[:, 0] = x_pred_n.detach().clone()
        y_pred[:, 0] = y_pred_n

        # loop over times
        for n in range(len(times)):
            # perform the filtering/assimilation step w/ constant gain K

            x_assim_n = x_pred_n + (self.K @ (y_obs[:, n] - y_pred_n).T).T

            x_assim[:, n] = x_assim_n.detach().clone()
            y_assim[:, n] = self.h_obs(x_assim_n.detach()).detach().clone()

            if n < y_obs.shape[1] - 1:
                # predict the next state by solving the ODE from t_n to t_{n+1}
                x_pred_n = self.solve(
                    x_assim_n, times[n : n + 2], params=self.odeint_params
                )[-1]
                x_pred[:, n + 1] = x_pred_n.detach().clone()

                # compute the observation map
                y_pred_n = self.h_obs(x_pred_n.clone())
                y_pred[:, n + 1] = y_pred_n.clone()

        return y_pred, y_assim, x_pred, x_assim, self.cov


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_hidden_layers=1,
        layer_width=50,
        activations="gelu",
        dropout=0.01,
    ):
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
            layers.append(nn.Linear(layer_width[i - 1], layer_width[i]))
            layers.append(get_activation(self.activations[i]))
            layers.append(nn.Dropout(p=dropout))  # Dropout layer added here

        # Output layer
        layers.append(nn.Linear(layer_width[-1], output_size))

        # Sequentially stack the layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class F_Physics(nn.Module):
    def __init__(self, ode=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ode = ode

    def forward(self, x):
        return self.ode.rhs(0, x)


# Define a class for the learned ODE model
# This has a forward method to represent a RHS of an ODE, where rhs = f_physics + f_nn
class HybridODE(nn.Module):
    def __init__(
        self,
        dim_state,
        ode: object = None,
        use_physics: bool = False,
        use_nn: bool = True,
        num_hidden_layers=1,
        layer_width=50,
        activations="gelu",
        dropout=0.01,
    ):
        super(HybridODE, self).__init__()
        self.use_physics = use_physics
        self.use_nn = use_nn

        if self.use_physics:
            self.f_physics = F_Physics(ode)  # currently just a placeholder (outputs 0)

        if self.use_nn:
            self.f_nn = FeedForwardNN(
                dim_state,
                dim_state,
                num_hidden_layers,
                layer_width,
                activations,
                dropout,
            )

    def forward(self, t, x, bound=100):
        rhs = torch.zeros_like(x, requires_grad=True).to(x)

        # if x is outside of [-100,100], set rhs to 0 to achieve fixed point at a boundary instead of blow-up
        if torch.any(torch.abs(x) > bound):
            return rhs
        else:
            if self.use_physics:
                rhs = rhs + self.f_physics(x)
            if self.use_nn:
                rhs = rhs + self.f_nn(x)
        return rhs
