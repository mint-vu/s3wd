import torch
import torch.nn as nn
import numpy as np

def get_stereo_proj(x):
    d = x.shape[-1] - 1
    numerator = x[..., :d]
    denominator = 1 - x[..., d]
    near_pole = np.isclose(denominator, 0)
    proj = np.full_like(x[..., :d], np.inf) 
    proj[~near_pole] = numerator[~near_pole] / denominator[~near_pole, np.newaxis]
    return torch.tensor(proj)

class Phi(nn.Module):
    def __init__(self, size):
        super(Phi, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))
    def forward(self, x):
        xhat = self.net(x)
        return torch.cat((x,xhat),dim=-1)