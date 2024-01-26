import torch
import ot
import numpy as np

def g_wasserstein(Xs, Xt, p=1, device='cpu'):
    '''wasserstein with spherical geodesic cost'''
    ip = Xs @ Xt.T
    M = torch.arccos(torch.clamp(ip, min=-1+1e-5, max=1-1e-5)) ** p
    M = M.detach().cpu().numpy()
    a, b = np.ones((Xs.shape[0],)) / Xs.shape[0], np.ones((Xt.shape[0],)) / Xt.shape[0]
    dist = ot.emd2(a, b, M)
    return torch.tensor(dist, dtype=torch.float32, device=device)

def g_sinkhorn(Xs, Xt, reg=0.1, numItermax=1000, device='cpu'):
    '''sinkhorn with spherical geodesic cost'''
    ip = Xs @ Xt.T
    M = torch.arccos(torch.clamp(ip, min=-1+1e-5, max=1-1e-5))
    M = M.detach().cpu().numpy()  
    a, b = np.ones((Xs.shape[0],)) / Xs.shape[0], np.ones((Xt.shape[0],)) / Xt.shape[0]
    dist = ot.sinkhorn2(a, b, M, reg, numItermax=numItermax)
    return torch.tensor(dist, dtype=torch.float32, device=device)
