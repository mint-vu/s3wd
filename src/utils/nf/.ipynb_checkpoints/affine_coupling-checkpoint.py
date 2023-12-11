import torch.nn as nn
import torch

from utils.nf.base_nf import BaseNF

class AffineCoupling(BaseNF):
    """
    	Affine Coupling layer

    	Refs:
        - Density estimation using RealNVP
    """

    def __init__(self, scaling, shifting, dim):
        super().__init__()
        self.scaling = scaling
        self.shifting = shifting
        self.k = dim//2

    def forward(self, x):
        x0, x1 = x[:,:self.k], x[:,self.k:]

        s = self.scaling(x0)
        t = self.shifting(x0)
        z0 = x0
        z1 = torch.exp(s)*x1+t

        z = torch.cat([z0,z1], dim=1)
        return z, torch.sum(s, dim=1)


    def backward(self, z):
        z0, z1 = z[:,:self.k], z[:,self.k:]

        s = self.scaling(z0)
        t = self.shifting(z0)
        x0 = z0
        x1 = torch.exp(-s)*(z1-t)
        
        x = torch.cat([x0,x1], dim=1)
        return x #, -torch.sum(s, dim=1)