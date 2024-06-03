import torch
from utils.misc import generate_rand_projs, generate_equator_projs

def swd(X, Y, p, n_projs=200, device='cpu'):
    projs = generate_rand_projs(X.shape[-1], n_projs).to(device)
    s1_proj, s2_proj = X.double() @ projs.T, Y.double() @ projs.T

    d = torch.abs(torch.sort(s1_proj.transpose(0, 1), dim=1).values - 
                  torch.sort(s2_proj.transpose(0, 1), dim=1).values)

    wd = d.pow(p).sum(dim=1).mean()
    return wd

def vswd(X, Y, p, n_projs=200, device='cpu'):
    projs = generate_equator_projs(X.shape[-1], n_projs).to(device)
    s1_proj, s2_proj = X.double() @ projs.T, Y.double() @ projs.T
    d = torch.abs(torch.sort(s1_proj.transpose(0, 1), dim=1).values -
                  torch.sort(s2_proj.transpose(0, 1), dim=1).values)
    vswd = d.pow(p).sum(dim=1).mean()
    return vswd