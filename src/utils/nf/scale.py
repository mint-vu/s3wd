import torch.nn as nn
import torch

from utils.nf.base_nf import BaseNF

class Scale(BaseNF):
    """
    	Scaling layer

    	Refs:
    	- NICE: Non-linear independent components estimation
    """

    def __init__(self, dim):
        super().__init__()
        self.log_s = nn.Parameter(torch.randn(1, dim, requires_grad=True))

    def forward(self, x):
        return torch.exp(self.log_s)*x, torch.sum(self.log_s, dim=1)
    
    def backward(self, z):
        return torch.exp(-self.log_s)*z #, -torch.sum(self.log_s, dim=1)
