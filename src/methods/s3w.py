import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import generate_rand_projs
from utils.s3w_utils import get_stereo_proj_torch, Phi, calc_rot, rotate, hStar, make_virtual_grid, find_pnp

from tqdm import tqdm
import time

    
def s3wd(X, Y, p, h=None, n_projs=1000, device='cpu', random_pnp=True):
    if h is None: h = hStar()
    
    X_ = X.to(device)
    Y_ = Y.to(device)
    
    if random_pnp:
        pnp = torch.rand(3,device=device)
        pnp[-1] = 0
        pnp /= torch.norm(pnp)
    else:
        grid = make_virtual_grid(n_points=1000, device=device)
        pnp = find_pnp(torch.concat((X_, Y_), dim=0), grid, device=device)

    rot_axis, rot_angle = calc_rot(pnp, device=device)
    rot_axis = rot_axis.to(device)
    rot_angle = rot_angle.to(device)

    X_ = rotate(X_, rot_axis, rot_angle)
    Y_ = rotate(Y_, rot_axis, rot_angle)

    X_sp = get_stereo_proj_torch(X_).to(device)
    Y_sp = get_stereo_proj_torch(Y_).to(device)
    s1_h = h(X_sp).double()
    s2_h = h(Y_sp).double()

    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T

    d = torch.abs(torch.sort(s1_h_rp.transpose(0, 1), dim=1).values - 
                  torch.sort(s2_h_rp.transpose(0, 1), dim=1).values)

    wd = d.pow(p).sum(dim=1).pow(1. / p).mean()
    return wd

def max_s3wd(X, Y, n_projs, p=2, n_iters=1000, lam=20, device='cpu'):
    h = hPhi().to(device)
    h_optimizer = optim.Adam(h.parameters(), lr=8e-3, betas=(0.999, 0.999))
    
    X_sp = get_stereo_proj(X.detach()).to(device)
    Y_sp = get_stereo_proj(Y.detach()).to(device)
    
    for _ in range(n_iters):
        h_optimizer.zero_grad()
        
        s1_h, s2_h = h(X_sp).double(), h(Y_sp).double()
        
        reg = lam * (s1_h.norm(p=2, dim=1) + s2_h.norm(p=2, dim=1)).mean()
        projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
        s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
        
        d = torch.abs(torch.sort(s1_h_rp.T, dim=1).values - torch.sort(s2_h_rp.T, dim=1).values)
        wd = torch.sum(torch.pow(d, p), dim=1)
        wd = torch.pow(wd.mean() + 1e-12, 1. / p)  
        
        loss = reg - wd

        loss.backward(retain_graph=True)
        h_optimizer.step()
    
    s1_h, s2_h = h(X_sp), h(Y_sp)
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.T, dim=1).values - torch.sort(s2_h_rp.T, dim=1).values)
    wd = torch.sum(torch.pow(d, p), dim=1)
    wd = torch.pow(wd.mean() + 1e-12, 1. / p) 

    return wd