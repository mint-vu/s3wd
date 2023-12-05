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

class Phi(nn.Module):
    def __init__(self, size):
        super(Phi, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))
    def forward(self, x):
        xhat = self.net(x)
        return torch.cat((x,xhat),dim=-1)
    
    
class LogMap(nn.Module):
    def __init__(self, size):
        super(LogMap, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size), nn.ReLU())

    def forward(self, x):
        xhat = self.net(x)
        xhat_t = torch.log(xhat + 1e-8)
        # xhat_t = torch.log(xhat + 1e-8)
        # return torch.cat((x, xhat_t), dim=1)
        return torch.cat((x,xhat_t),dim=-1)