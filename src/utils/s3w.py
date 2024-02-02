import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import geotorch


class RotationPool:
    rot_matrices_ = None
    d_ = None
    pool_size_ = None

    @staticmethod
    def get(d, pool_size, device='cpu'):
        if RotationPool.rot_matrices_ is None or RotationPool.d_ != d or RotationPool.pool_size_ != pool_size:
            RotationPool.rot_matrices_ = torch.stack([geotorch.SO(torch.Size([d, d])).sample('uniform') for _ in range(pool_size)]).to(device)
            RotationPool.d_ = d
            RotationPool.pool_size_ = pool_size
        return RotationPool.rot_matrices_
    
    @staticmethod
    def reset():
        RotationPool.rot_matrices_ = None
        RotationPool.d_ = None
        RotationPool.pool_size_ = None
        
class RotationSchedule:
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.current_epoch = 0
        self.current_n_rotations = 0

    def step(self):
        if self.current_epoch < self.n_epochs - 1:
            self.current_epoch += 1
            self.update()
        else:
            self.current_n_rotations = self.get_max()

    def update(self):
        raise NotImplementedError

    def get(self):
        return self.current_n_rotations

    def get_max(self):
        return self.current_n_rotations

class LinearRS(RotationSchedule):
    def __init__(self, min_n_rotations, max_n_rotations, n_epochs):
        super().__init__(n_epochs)
        self.min_n_rotations = min_n_rotations
        self.max_n_rotations = max_n_rotations
        self.step_size = (max_n_rotations - min_n_rotations) / n_epochs
        self.current_n_rotations = self.min_n_rotations
    
    def update(self):
        n_rotations = self.min_n_rotations + self.step_size * self.current_epoch
        self.current_n_rotations = min(int(n_rotations), self.max_n_rotations)

class CustomRS(RotationSchedule):
    def __init__(self, schedule_dict, n_epochs):
        super().__init__(n_epochs)
        self.schedule_dict = schedule_dict
        self.update()

    def update(self):
        self.current_n_rotations = self.schedule_dict.get(self.current_epoch, self.current_n_rotations)

    def get_max(self):
        return self.schedule_dict.get(self.n_epochs - 1, self.current_n_rotations)

def unif_hypersphere(shape, device):
    samples = torch.randn(shape, device=device)
    samples = F.normalize(samples, p=2, dim=-1)
    return samples

def get_stereo_proj(x):
    d = x.shape[-1] - 1
    numerator = x[..., :d]
    denominator = 1 - x[..., d]
    near_pole = np.isclose(denominator, 0, atol=1e-6)
    proj = np.full_like(x[..., :d], np.inf) 
    proj[~near_pole] = numerator[~near_pole] / denominator[~near_pole, np.newaxis]
    return torch.tensor(proj)

def get_stereo_proj_torch(x, epsilon=1e-6):
    d = x.shape[-1] - 1
    numerator = 2 * x[..., :d]
    denominator = 1 - x[..., d]
    near_pole = torch.isclose(denominator, torch.zeros_like(denominator), atol=epsilon)
    proj = torch.full_like(x[..., :d], float('inf')) 
    proj[~near_pole] = numerator[~near_pole] / denominator[~near_pole].unsqueeze(-1)
    return proj

def epsilon_projection(x, epsilon=1e-6):
    n = torch.where(x[..., -1] == 1.)
    if n[0].numel() > 0:    
        x.data[n] = x[n] + (epsilon * torch.rand_like(x[n]) - epsilon/2)
    x.data[..., -1] = torch.min(x[..., -1], torch.tensor(1.-epsilon)) 
    alpha = torch.sqrt((1 - x[..., -1]**2)/(x[..., :-1]**2).sum(-1))
    alpha[alpha.isnan()] = 1. # Correct for instances where the point is at the south pole
    x.data[..., :-1] *= alpha.unsqueeze(-1)  
    return x

class hStar(nn.Module):
    # h1 in the paper
    def __init__(self):
        super(hStar, self).__init__()

    def forward(self, x, epsilon=1e-6):
        x_sq_sum = (x**2).sum(-1).clamp(min=1e-8)
        x_dp1 = torch.clip((x_sq_sum - 1) / (x_sq_sum + 1), -1, 1)
        arc_input = torch.clip(-x_dp1, -1 + epsilon, 1 - epsilon)
        arccos = torch.arccos(arc_input)
        h = (arccos / np.pi).unsqueeze(-1) * (x / torch.sqrt(x_sq_sum).unsqueeze(-1))
        return h