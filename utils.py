import numpy as np
from scipy.stats import gaussian_kde
import itertools
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)


def batch_covariance(ensemble):
    # ensemble shape: (N_batch, N_ensemble, dim_state)
    N_batch, N_ensemble, dim_state = ensemble.shape

    # Compute mean for each batch
    mean = torch.mean(ensemble, dim=1, keepdim=True)

    # Compute deviations from the mean
    deviations = ensemble - mean

    # Calculate covariance matrix for each batch
    covariance_matrices = torch.matmul(deviations.transpose(-2, -1), deviations) / (
        N_ensemble - 1
    )

    return covariance_matrices


# define a weighted mean squared error loss where the weights are the inverse of a NxN covariance matrix
def weighted_mse_loss(input, target, inv_cov):
    # weight is a K x N x N matrix, where K is length of input/target and N is the dimension of the system
    # apply the kth weight to the kth input/target pair
    # TODO: double check that this is righ!
    return torch.mean(inv_cov * (input - target).permute(1, 0, 2) ** 2)


def log_det(cov):
    # cov is a K x N x N matrix, where K is length of input/target and N is the dimension of the system
    # record the sum of the log determinants of the covariance matrices
    # TODO: double check that this is righ!
    return torch.mean(torch.log(torch.det(cov)))


def neg_log_likelihood_loss(input, target, cov, inv_cov):
    return weighted_mse_loss(input, target, inv_cov) + log_det(cov)


def discretized_univariate_kde(x, n_eval_bins=100):
    """Returns a discretized univariate kernel density estimate of the data."""
    kde = gaussian_kde(x)
    x_eval = np.linspace(np.min(x), np.max(x), n_eval_bins)
    return x_eval, kde(x_eval)


def odeint_wrapper(
    func,
    y0,
    t,
    use_adjoint=False,
    rtol=1e-7,
    atol=1e-9,
    method=None,
    options=None,
    event_fn=None,
):
    """Wrapper for odeint and odeint_adjoint to allow for easy switching between the two.
    Uses default values for rtol, atol, method, and options if not specified."""

    if use_adjoint:
        return odeint_adjoint(
            func,
            y0,
            t,
            rtol=rtol,
            atol=atol,
            method=method,
            options=options,
            event_fn=event_fn,
        )
    else:
        return odeint(
            func,
            y0,
            t,
            rtol=rtol,
            atol=atol,
            method=method,
            options=options,
            event_fn=event_fn,
        )


def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "gelu":
        return nn.GELU()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


def dict_combiner(mydict):
    if mydict:
        keys, values = zip(*mydict.items())
        experiment_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        experiment_list = [{}]
    return experiment_list


### Normalizers: MaxMin, UnitGaussian, Inactive ###
class MaxMinNormalizer(object):
    def __init__(self, x, eps=1e-5):
        super(MaxMinNormalizer, self).__init__()

        self.max = np.max(x, 0)
        self.min = np.min(x, 0)
        self.range = self.max - self.min
        self.eps = eps

    def encode(self, x):
        return (x - self.min) / (self.range + self.eps)

    def decode(self, x):
        return self.min + x * (self.range + self.eps)


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = np.mean(x, 0)
        self.std = np.std(x, 0)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return (x * (self.std + self.eps)) + self.mean


class InactiveNormalizer(object):
    def __init__(self, x, eps=1e-5):
        super(InactiveNormalizer, self).__init__()

    def encode(self, x):
        return x

    def decode(self, x):
        return x
