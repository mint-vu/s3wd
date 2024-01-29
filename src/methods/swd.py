import torch
from utils.misc import generate_rand_projs

def swd(samples_1, samples_2, p, n_projs=200, device='cpu'):
    projs = generate_rand_projs(samples_1.shape[-1], n_projs).to(device)
    s1_proj, s2_proj = samples_1.double() @ projs.T, samples_2.double() @ projs.T

    d = torch.abs(torch.sort(s1_proj.transpose(0, 1), dim=1).values - 
                  torch.sort(s2_proj.transpose(0, 1), dim=1).values)

    wd = d.pow(p).sum(dim=1).mean()
    return wd