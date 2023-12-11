import torch.nn as nn
import torch

from utils.nf.base_nf import BaseNF

class RadialFlow(BaseNF):
    """
    	Refs
        - Variational Inference with Normalizing Flows, https://arxiv.org/pdf/1505.05770.pdf
    """
    def __init__(self, dim):
        super().__init__()
        self.d = dim
        self.log_alpha = nn.Parameter(torch.randn(1, requires_grad=True))
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True))
        self.z0 = nn.Parameter(torch.randn(dim, requires_grad=True))   

    def forward(self, x):
        r = torch.norm(x-self.z0,dim=-1,keepdim=True)

        alpha = torch.exp(self.log_alpha)
        h = 1/(alpha+r)
        beta = -alpha+torch.log(1+torch.exp(self.beta))

        z = x+beta*h*(x-self.z0)
        
        log_det = (self.d-1)*torch.log(1+beta*h)+torch.log(1+beta*h-beta*r/(alpha+r)**2)

        return z, log_det[:,0]
    
    def backward(self, z):
        raise NotImplementerError
    