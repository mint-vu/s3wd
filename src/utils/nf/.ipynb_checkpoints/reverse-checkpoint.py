import torch

from utils.nf.base_nf import BaseNF

class Reverse(BaseNF):
    """
    	Reverse the indices

        Refs:
        - https://github.com/acids-ircam/pytorch_flows/blob/master/flows_04.ipynb
    """

    def __init__(self, dim):
        super().__init__()
        self.permute = torch.arange(dim-1, -1, -1)
        self.inverse = torch.argsort(self.permute)

    def forward(self, x):
        return x[:, self.permute], torch.zeros(x.size(0))

    def backward(self, z):
        return z[:, self.inverse] #, torch.zeros(z.size(0), device='cpu')


class Shuffle(Reverse):
    """
        Apply a random permutation of the indices
    """
    def __init__(self, d):
        super().__init__(d)
        self.permute = torch.randperm(d)
        self.inverse = torch.argsort(self.permute)
        
