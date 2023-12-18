import torch
import torch.nn as nn

from utils.nf.exp_map import ExpMap
from utils.nf.base_nf import BaseNF
from utils.nf.additive_coupling import AdditiveCoupling


class NormalizingFlows(BaseNF):
    """
        Composition of flows
        
        Refs: 
        - https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    """
    def __init__(self, flows, device):
        """
    		Inputs:
    		- flows: list of BaseNFs objects
    	"""
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.device = device
        
    def forward(self, x):
        log_det = torch.zeros(x.shape[0]).to(self.device)
        zs = [x]
        for flow in self.flows:
            x, log_det_i = flow(x)
            log_det += log_det_i
            zs.append(x)
        return zs, log_det
    
    def backward(self, z):
        log_det = torch.zeros(z.shape[0])
        xs = [z]
        for flow in self.flows[::-1]:
            z = flow.backward(z)
            xs.append(z)
        return xs
        

def make_NF(d=3, n_blocks=6, n_components=5, device='cpu'):
    flows = []
    for k in range(n_blocks):
        radialBlock = ExpMap(d, n_components).to(device)
        flows.append(radialBlock)

    model = NormalizingFlows(flows, device=device).to(device)
    return model
