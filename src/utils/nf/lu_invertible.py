import torch.nn as nn
import torch

from utils.nf.base_nf import BaseNF

class LUInvertible(BaseNF):
    """
    	Invertible 1x1 convolution (based on LU decomposition)
    	
    	Refs:
    	- Glow: Generative flow with invertible 1×1 convolutions
        - https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
        - https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.W = torch.Tensor(dim, dim)
        nn.init.orthogonal_(self.W)

        # P, L, U = torch.lu_unpack(*self.W.lu())
        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.S = nn.Parameter(torch.from_numpy(U).diag())
        self.U = nn.Parameter(torch.triu(torch.from_numpy(U),1))

    def forward(self, x):
        P = self.P.to(device)
        L = torch.tril(self.L,-1)+torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U,1)+torch.diag(self.S)
        W = P @ L @ U
        return x@W, torch.sum(torch.log(torch.abs(self.S)))

    def backward(self, z):
        P = self.P.to(device)
        L = torch.tril(self.L,-1)+torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U,1)+torch.diag(self.S)
        W = P @ L @ U
        return z@torch.inverse(W) #, -torch.sum(torch.log(torch.abs(self.S)))


