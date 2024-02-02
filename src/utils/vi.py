import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from tqdm.auto import trange
from copy import deepcopy

from methods.sswd import sswd
# from methods.sw import swd
from utils.power_spherical import PowerSpherical
from utils.nf.normalizing_flows import make_NF
from utils.plot import scatter_mollweide, plot_target_density
from utils.misc import spherical_to_euclidean_torch, spherical_to_euclidean

def ULA_sphere(V, n_particles=1000, d=2, dt=1e-3, n_steps=4000, device='cpu', init_particles=None):
    normal = D.MultivariateNormal(torch.zeros(d, device=device), torch.eye(d, device=device))

    if init_particles is None:
        x0 = normal.sample((n_particles,))
        x0 = F.normalize(x0, p=2, dim=-1)  
    else:
        x0 = init_particles

    xk = x0.clone()
    trajectory = [x0.clone().detach().cpu()]

    pbar = trange(n_steps) 

    for k in pbar:
        xk.requires_grad_(True)
        grad_V = torch.autograd.grad(V(xk).sum(), xk)[0]
        W = normal.sample((n_particles,))
        xk = xk.detach()

        v = -grad_V * dt + np.sqrt(2 * dt) * W
        v = v - torch.sum(v * xk, axis=-1)[:, None] * xk  
        norm_v = torch.linalg.norm(v, axis=-1)[:, None]

        xk = xk * torch.cos(norm_v) + torch.sin(norm_v) * v / norm_v
        trajectory.append(xk.clone().detach().cpu())

    return xk, trajectory

def kl_ess(log_model_prob, target_prob):
    weights = target_prob / np.exp(log_model_prob)
    Z = np.mean(weights)
    KL = np.mean(log_model_prob - np.log(target_prob)) + np.log(Z)
    ESS = np.sum(weights) ** 2 / np.sum(weights ** 2)
    return Z, KL, ESS

def target_density(x):
    m = torch.matmul(x, target_mus.T)
    return torch.sum(torch.exp(10 * m), dim=-1)

target_mu = spherical_to_euclidean(np.array([
    [1.5, 0.7 + np.pi / 2],
    [1., -1. + np.pi / 2],
    [5., 0.6 + np.pi / 2],  # 0.5 -> 5.!
    [4., -0.7 + np.pi / 2]
]))

def s2_target(x):
    xe = np.dot(x, target_mu.T)
    return np.sum(np.exp(10 * xe), axis=1)



