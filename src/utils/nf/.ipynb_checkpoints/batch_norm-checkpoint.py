import torch.nn as nn
import torch

from utils.nf.base_nf import BaseNF

class BatchNorm(BaseNF):
    """
        Refs: 
        - https://github.com/acids-ircam/pytorch_flows/blob/master/flows_04.ipynb
        - Masked Autoregressive Flows for Density Estimation 
        - Density Estimation Using Real NVP
    """
    def __init__(self, dim, eps=1e-5, momentum=0.95):
        super().__init__()
        self.eps = eps
        self.momentum = momentum ## To compute train set mean
        self.train_mean = torch.zeros(dim)
        self.train_var = torch.ones(dim)

        self.gamma = nn.Parameter(torch.ones(dim, requires_grad=True))
        self.beta = nn.Parameter(torch.ones(dim, requires_grad=True))

    def forward(self, x):
        """
            mean=batch_mean in training time, mean of the entire dataset in test
        """
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = (x-x.mean(0)).pow(2).mean(0)+self.eps

            self.train_mean = self.momentum*self.train_mean+(1-self.momentum)*self.batch_mean
            self.train_var = self.momentum*self.train_var+(1-self.momentum)*self.batch_var

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.train_mean
            var = self.train_var

        z = torch.exp(self.gamma)*(x-mean)/var.sqrt()+self.beta
        log_det = torch.sum(self.gamma-0.5*torch.log(var))
        return z, log_det

    def backward(self, z):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.train_mean
            var = self.train_var

        x = (z-self.beta)*torch.exp(-self.gamma)*var.sqrt()+mean
        log_det = torch.sum(-self.gamma+0.5*torch.log(var))
        return x #, log_det
        