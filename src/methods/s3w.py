import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import generate_rand_projs
from utils.s3w_utils import get_stereo_proj_torch, Phi

    
def s3wd(samples_1, samples_2, p, h=None, n_projs=1000, device='cpu'):
    if h is None: h=nn.Identity()
    samples_1_sp = get_stereo_proj_torch(samples_1).to(device)
    samples_2_sp = get_stereo_proj_torch(samples_2).to(device)

    s1_h = h(samples_1_sp).double()
    s2_h = h(samples_2_sp).double()
    
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T    

    d = torch.abs(torch.sort(s1_h_rp.transpose(0, 1), dim=1).values - 
            torch.sort(s2_h_rp.transpose(0, 1), dim=1).values)

    wd = d.pow(p).sum(dim=1).pow(1. / p).mean()
    
    return wd

def max_s3wd(samples_1, samples_2, n_projs, p=2, n_iters=1000, lam=20, device='cpu'):
    h = Phi(samples_1.size(1)-1).to(device)
    h_optimizer = optim.Adam(h.parameters(), lr=8e-3, betas=(0.999, 0.999))
    
    samples_1_sp = get_stereo_proj(samples_1.detach()).to(device)
    samples_2_sp = get_stereo_proj(samples_2.detach()).to(device)
    
    for _ in range(n_iters):
        h_optimizer.zero_grad()
        
        s1_h, s2_h = h(samples_1_sp).double(), h(samples_2_sp).double()
        
        reg = lam * (s1_h.norm(p=2, dim=1) + s2_h.norm(p=2, dim=1)).mean()
        projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
        s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
        
        d = torch.abs(torch.sort(s1_h_rp.T, dim=1).values - torch.sort(s2_h_rp.T, dim=1).values)
        wd = torch.sum(torch.pow(d, p), dim=1)
        wd = torch.pow(wd.mean() + 1e-12, 1. / p)  
        
        loss = reg - wd

        loss.backward(retain_graph=True)
        h_optimizer.step()
    
    s1_h, s2_h = h(samples_1_sp), h(samples_2_sp)
    projs = generate_rand_projs(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.T, dim=1).values - torch.sort(s2_h_rp.T, dim=1).values)
    wd = torch.sum(torch.pow(d, p), dim=1)
    wd = torch.pow(wd.mean() + 1e-12, 1. / p) 

    return wd