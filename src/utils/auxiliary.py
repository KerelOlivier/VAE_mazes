"""
Adapted from Jakub Tomczak's repository:
https://github.com/jmtomczak/intro_dgm
"""

import torch
import torch.nn.functional as F
import numpy as np

# DO NOT REMOVE
PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-5

def log_categorical(x, p, num_classes=256, reduction=None):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'mean':
        return torch.mean(log_p, list(range(1, len(x.shape))))
    elif reduction == 'sum':
        return torch.sum(log_p, list(range(1, len(x.shape))))
    else:
        return log_p

def log_bernoulli(x, p, reduction=None):
    pp = torch.clamp(p, EPS, 1. - EPS)
    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)
    if reduction == 'mean':
        return torch.mean(log_p, list(range(1, len(x.shape))))
    elif reduction == 'sum':
        return torch.sum(log_p, list(range(1, len(x.shape))))
    else:
        return log_p

def log_normal_diag(x, mu, log_var, reduction=None):
    D = x.shape[1]
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'mean':
        return torch.mean(torch.sum(log_p, list(range(1, len(x.shape)))))
    elif reduction == 'sum':
        return torch.sum(torch.sum(log_p, list(range(1, len(x.shape)))))
    else:
        return log_p


def log_standard_normal(x, reduction=None):
    D = x.shape[1]
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'mean':
        return torch.mean(log_p, list(range(1, len(x.shape))))
    elif reduction == 'sum':
        return torch.sum(log_p, list(range(1, len(x.shape))))
    else:
        return log_p