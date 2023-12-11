import torch.nn as nn
import torch

from utils.nf.base_nf import BaseNF

class AdditiveCoupling(BaseNF):
    """
    	Additive coupling layer

    	Refs:
    	- NICE: Non-linear independent components estimation
    """

    def __init__(self, coupling, dim):
        super().__init__()
        self.k = dim//2
        self.coupling = coupling

    def forward(self, x):
        x0, x1 = x[:,:self.k], x[:,self.k:]
        
        m = self.coupling(x0)
        z0 = x0
        z1 = x1+m
            
        z = torch.cat([z0,z1], dim=1)
        return z,torch.zeros(x.shape[0])
    
    def backward(self, z):
        z0, z1 = z[:,:self.k], z[:,self.k:]

        m = self.coupling(z0)
        x0 = z0
        x1 = z1-m

        x = torch.cat([x0,x1], dim=1)
        return x #, torch.zeros(z.shape[0],device='cpu')
