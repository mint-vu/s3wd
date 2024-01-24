import torch
import torch.nn as nn
import torch.optim as optim
import geotorch
import numpy as np

from utils.utils import generate_rand_projs
from utils.s3w_utils import get_stereo_proj_torch, epsilon_projection, Phi, hStar, RotationPool

from typing import List


def ri_s3wd(X, Y, p, h=None, n_projs=1000, n_rotations=1, device='cpu'):
    # NOTE: h must accept vectors of the form (n_rotations, n_points, dim)
    if h is None: h = hStar()
    
    X = X.to(device)
    Y = Y.to(device)
    
    n = X.shape[-1]
    rot_matrices = [geotorch.SO(torch.Size([n, n])).sample('uniform') for _ in range(n_rotations)]
    rot_matrices = torch.stack(rot_matrices).to(device)
    X_rot = (rot_matrices @ X.T).permute(0, 2, 1)
    Y_rot = (rot_matrices @ Y.T).permute(0, 2, 1)

    X_eps = epsilon_projection(X_rot)
    Y_eps = epsilon_projection(Y_rot)

    X_sp = get_stereo_proj_torch(X_eps).to(device)
    Y_sp = get_stereo_proj_torch(Y_eps).to(device)

    s1_h = h(X_sp).double()
    s2_h = h(Y_sp).double()

    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T

    d = torch.abs(torch.sort(s1_h_rp.transpose(-2, -1), dim=-1).values - 
                  torch.sort(s2_h_rp.transpose(-2, -1), dim=-1).values)

    wd = d.pow(p).sum(dim=-1).mean(dim=-1)
    return wd.mean()

def ri_s3wd_unif(X, p, h=None, n_projs=1000, n_rotations=1, device='cpu'):
    # NOTE: h must accept vectors of the form (n_rotations, n_points, dim)
    if h is None: h = hStar()
    
    X = X.to(device)
    Y_unif = rand_u_hypersphere((n_rotations, X.shape[0], n), n, device=device)
    
    n = X.shape[-1]
    rot_matrices = [geotorch.SO(torch.Size([n, n])).sample('uniform') for _ in range(n_rotations)]
    rot_matrices = torch.stack(rot_matrices).to(device)
    X_rot = (rot_matrices @ X.T).permute(0, 2, 1)
    

    X_eps = epsilon_projection(X_rot)

    X_sp = get_stereo_proj_torch(X_eps).to(device)
    Y_unif_sp = get_stereo_proj_torch(epsilon_projection(Y_unif)).to(device)

    s1_h = h(X_sp).double()
    s2_h = h(Y_unif_sp).double()

    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T

    d = torch.abs(torch.sort(s1_h_rp.transpose(-2, -1), dim=-1).values - 
                  torch.sort(s2_h_rp.transpose(-2, -1), dim=-1).values)

    wd = d.pow(p).sum(dim=-1).mean(dim=-1)
    return wd.mean()

def ari_s3wd(X, Y, p, h=None, n_projs=1000, n_rotations=1, pool_size=100, device='cpu'):
    if h is None: h = hStar()

    n = X.shape[-1]

    assert pool_size >= n_rotations

    rotation_pool = RotationPool.generate(n, pool_size)

    indices = torch.randperm(rotation_pool.size(0))[:n_rotations]
    rot_matrices = rotation_pool[indices].to(device)

    X = X.to(device)
    Y = Y.to(device)
    
    X_rot = (rot_matrices @ X.T).permute(0, 2, 1)
    Y_rot = (rot_matrices @ Y.T).permute(0, 2, 1)

    X_eps = epsilon_projection(X_rot)
    Y_eps = epsilon_projection(Y_rot)

    X_sp = get_stereo_proj_torch(X_eps).to(device)
    Y_sp = get_stereo_proj_torch(Y_eps).to(device)

    s1_h = h(X_sp).double()
    s2_h = h(Y_sp).double()

    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T

    d = torch.abs(torch.sort(s1_h_rp.transpose(-2, -1), dim=-1).values - 
                  torch.sort(s2_h_rp.transpose(-2, -1), dim=-1).values)

    wd = d.pow(p).sum(dim=-1).mean(dim=-1)
    return wd.mean()

def ari_s3wd_unif(X, p, h=None, n_projs=1000, n_rotations=1, pool_size=100, device='cpu'):
    if h is None: h = hStar()

    n = X.shape[-1]

    assert pool_size >= n_rotations

    rotation_pool = RotationPool.generate(n, pool_size)

    indices = torch.randperm(rotation_pool.size(0))[:n_rotations]
    rot_matrices = rotation_pool[indices].to(device)

    X = X.to(device)
    Y_unif = rand_u_hypersphere((n_rotations, X.shape[0], n), n, device=device)
    
    X_rot = (rot_matrices @ X.T).permute(0, 2, 1)

    X_eps = epsilon_projection(X_rot)

    X_sp = get_stereo_proj_torch(X_eps).to(device)
    Y_unif_sp = get_stereo_proj_torch(epsilon_projection(Y_unif)).to(device)

    s1_h = h(X_sp).double()
    s2_h = h(Y_unif_sp).double()

    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T

    d = torch.abs(torch.sort(s1_h_rp.transpose(-2, -1), dim=-1).values - 
                  torch.sort(s2_h_rp.transpose(-2, -1), dim=-1).values)

    wd = d.pow(p).sum(dim=-1).mean(dim=-1)
    return wd.mean()
    
def s3wd(X, Y, p, h=None, n_projs=1000, device='cpu'):
    if h is None: h = hStar()
    
    X = X.to(device)
    Y = Y.to(device)
    
    X_eps = epsilon_projection(X)
    Y_eps = epsilon_projection(Y)

    X_sp = get_stereo_proj_torch(X_eps).to(device)
    Y_sp = get_stereo_proj_torch(Y_eps).to(device)
    s1_h = h(X_sp).double()
    s2_h = h(Y_sp).double()

    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T

    d = torch.abs(torch.sort(s1_h_rp.transpose(0, 1), dim=1).values - 
                  torch.sort(s2_h_rp.transpose(0, 1), dim=1).values)

    wd = d.pow(p).sum(dim=1).mean()
    return wd

def s3wd_unif(X, p, h=None, n_projs=1000, device='cpu'):
    if h is None: h = hStar()

    X = X.to(device)
    n = X.shape[-1]

    X_eps = epsilon_projection(X)
    X_sp = get_stereo_proj_torch(X_eps).to(device)
    s1_h = h(X_sp).double()

    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp = s1_h @ projs.T

    Y_unif = rand_u_hypersphere(X.shape[0], n,device=device)
    Y_unif_sp = get_stereo_proj_torch(epsilon_projection(Y_unif)).to(device)
    s2_h = h(Y_unif_sp).double()

    s2_h_rp = s2_h @ projs.T

    d = torch.abs(torch.sort(s1_h_rp.transpose(0, 1), dim=1).values - 
                  torch.sort(s2_h_rp.transpose(0, 1), dim=1).values)

    wd = d.pow(p).sum(dim=1).pow(1. / p).mean()
    return wd