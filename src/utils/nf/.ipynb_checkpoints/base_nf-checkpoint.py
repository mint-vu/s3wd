from abc import ABC, abstractmethod
import torch.nn as nn

class BaseNF(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, z):
        pass
