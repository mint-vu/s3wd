import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import generate_rand_projections
from utils.s3w_utils import get_stereo_projection, Phi

    
def s3wd(samples_1, samples_2, p, n_projs=1000, device='cpu'):
    samples_1_sp = get_stereo_projection(samples_1).to(device)
    samples_2_sp = get_stereo_projection(samples_2).to(device)

    projs = generate_rand_projections(samples_1_sp.shape[-1], n_projs).to(device)
    samples_1_rp, samples_2_rp = samples_1_sp @ projs.T, samples_2_sp @ projs.T    
    d = torch.abs(torch.sort(samples_1_rp.T, dim=1).values - torch.sort(samples_2_rp.T, dim=1).values)
    wd = d.pow(p).sum(dim=1).pow(1. / p).mean()
    
    return wd

def max_s3wd(samples_1, samples_2, n_projs, p=2, n_iters=100, lam=20, device='cpu'):
    h = Phi(samples_1.size(1)-1).to(device)
    h_optimizer = optim.Adam(h.parameters(), lr=8e-4, betas=(0.999, 0.999))
    
    samples_1_sp = get_stereo_projection(samples_1).to(device)
    samples_2_sp = get_stereo_projection(samples_2).to(device)
    
    for _ in range(n_iters):
        h_optimizer.zero_grad()
        
        s1_h, s2_h = h(samples_1_sp), h(samples_2_sp)
        
        reg = lam * (s1_h.norm(p=2, dim=1) + s2_h.norm(p=2, dim=1)).mean()
        projs = generate_rand_projections(s1_h.shape[-1], n_projs).to(device)
        s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
        
        d = torch.abs(torch.sort(s1_h_rp.T, dim=1).values - torch.sort(s2_h_rp.T, dim=1).values)
        wd = (d.pow(p).sum(dim=1) * 512 / samples_1.shape[0] + 1e-12).pow(1. / p).mean()
        
        loss = reg - wd

        loss.backward(retain_graph=True)
        h_optimizer.step()
    
    s1_h, s2_h = h(samples_1_sp), h(samples_2_sp)
    projs = generate_rand_projections(s1_h.shape[-1], n_projs).to(device)
    s1_h_rp, s2_h_rp = s1_h @ projs.T, s2_h @ projs.T
    d = torch.abs(torch.sort(s1_h_rp.T, dim=1).values - torch.sort(s2_h_rp.T, dim=1).values)
    wd = d.pow(p).sum(dim=1).pow(1. / p).mean()

    return wd

