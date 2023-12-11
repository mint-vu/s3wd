import torch
import torch.nn as nn
import numpy as np

def get_stereo_proj(x):
    d = x.shape[-1] - 1
    numerator = x[..., :d]
    denominator = 1 - x[..., d]
    near_pole = np.isclose(denominator, 0, rtol=1e-6, atol=1e-6)
    proj = np.full_like(x[..., :d], np.inf) 
    proj[~near_pole] = numerator[~near_pole] / denominator[~near_pole, np.newaxis]
    return torch.tensor(proj)

def get_stereo_proj_torch(x):
    d = x.shape[-1] - 1
    numerator = 2 * x[..., :d]
    denominator = 1 - x[..., d]
    near_pole = torch.isclose(denominator, torch.zeros_like(denominator), rtol=1e-6, atol=1e-6)
    proj = torch.full_like(x[..., :d], float('inf')) 
    proj[~near_pole] = numerator[~near_pole] / denominator[~near_pole].unsqueeze(-1)
    return proj

class Phi(nn.Module):
    def __init__(self, size):
        super(Phi, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))
    def forward(self, x):
        xhat = self.net(x)
        return torch.cat((x,xhat),dim=-1)