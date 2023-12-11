import torch.nn as nn
import torch

from utils.nf.base_nf import BaseNF

class PlanarFlow(BaseNF):
    """
    	Refs
        - Variational Inference with Normalizing Flows, https://arxiv.org/pdf/1505.05770.pdf
    """
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim, requires_grad=True))
        self.w = nn.Parameter(torch.randn(1, dim, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, 1, requires_grad=True))

    def forward(self, x):
        # enforce invertibility
        wu = self.w@self.u.t()
        m_wu = -1+torch.log(1+torch.exp(wu))
        u_hat = self.u+(m_wu-wu)*self.w/torch.sum(self.w**2)

        z = x+u_hat*torch.tanh(x@self.w.t()+self.b)
        psi = (1-torch.pow(torch.tanh(x@self.w.t()+self.b),2))*self.w
        log_det = torch.log(1+psi@u_hat.t())
        return z, log_det[:,0]

    def backward(self, z):
        # can't compute it analytically
        return NotImplementedError
        
        