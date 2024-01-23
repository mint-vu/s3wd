import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import geotorch

<<<<<<< HEAD

class RotationPool:
    rot_matrices_ = None
    d_ = None
    pool_size_ = None

    @staticmethod
    def generate(d, pool_size):
        if RotationPool.rot_matrices_ is None or RotationPool.d_ != d or RotationPool.pool_size_ != pool_size:
            RotationPool.rot_matrices_ = torch.stack([geotorch.SO(torch.Size([d, d])).sample('uniform') for _ in range(pool_size)])
            RotationPool.d_ = d
            RotationPool.pool_size_ = pool_size
        return RotationPool.rot_matrices_
=======
class RotationPool:
    rot_matrices = None
    d = None

    @staticmethod
    def generate(d, pool_size):
        if RotationPool.rot_matrices is None or RotationPool.d != d:
            RotationPool.rot_matrices = torch.stack([geotorch.SO(torch.Size([d, d])).sample('uniform') for _ in range(pool_size)])
            RotationPool.d = d
        return RotationPool.rot_matrices
    
>>>>>>> fec67281eee08c4342c17541c51d4960ec4b2e98
    @staticmethod
    def reset():
        RotationPool.rot_matrices = None
        RotationPool.d = None
<<<<<<< HEAD

=======
>>>>>>> fec67281eee08c4342c17541c51d4960ec4b2e98

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

class Phi(nn.Module):
    def __init__(self, size):
        super(Phi, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))
    def forward(self, x):
        xhat = self.net(x)
        return torch.cat((x,xhat),dim=-1)

class hStar(nn.Module):
    def __init__(self):
        super(hStar, self).__init__()

    def forward(self, x, epsilon=1e-6):
        x_sq_sum = (x**2).sum(-1).clamp(min=1e-8)
        x_dp1 = torch.clip((x_sq_sum - 1) / (x_sq_sum + 1), -1, 1)
        arc_input = torch.clip(-x_dp1, -1 + epsilon, 1 - epsilon)
        arccos = torch.arccos(arc_input)
        h = (arccos / np.pi).unsqueeze(-1) * (x / torch.sqrt(x_sq_sum).unsqueeze(-1))
        return h

class MLP(nn.Module):
    def __init__(self,architecture=[1,256,128,2],activation=nn.LeakyReLU()):
        super(MLP, self).__init__()
        self.activation = activation
        self.architecture = architecture
        arch = []
        for i in range(1, len(architecture)-1):
            arch.append(nn.Linear(architecture[i-1], architecture[i]))
            arch.append(self.activation)
        self.basis = nn.Sequential(*arch)
        self.regressor = nn.Linear(architecture[-2], architecture[-1])

    def forward(self,x):
        assert x.shape[1] == self.architecture[0]
        z = self.basis(x)
        out = self.regressor(z)
        return out
    
class FF(nn.Module):
    def __init__(self,din=3,dff=256,w0=30):
        super(FF, self).__init__()
        self.B = nn.Parameter(torch.randn(din, dff), requires_grad=False)
        self.w0 = w0
        self.dff_ = dff
        self.din_ = din
    def forward(self,x):
        z = x @ self.B
        return torch.cat([torch.sin(self.w0 * z), torch.cos(self.w0 * z)],1)
    
class hPhi(nn.Module):
    def __init__(self, input_dim=2, fourier_dff=512, mlp_architecture=[1024, 1024, 32], w0=30, device='cpu'):
        super(hPhi, self).__init__()
        self.FF = FF(din=input_dim, dff=fourier_dff, w0=w0)
        self.mlp = MLP(architecture=[fourier_dff * 2] + mlp_architecture)
        self.model=nn.Sequential(self.FF,self.mlp).to(device)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.model(x)

    def train(self, epochs, batchsize, lr=1e-4):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        lossE = []
        lossAvg = []
        
        pbar = tqdm(range(epochs), total=epochs)
        for epoch in pbar:
            z = torch.randn(batchsize, 3).to(self.device)
            z = z / (torch.norm(z, dim=1, keepdim=True))

            inner = (z.unsqueeze(1) * z.unsqueeze(0)).sum(2)
            inner = torch.clip(inner, -1, 1)
            arclengths_sq = (torch.arccos(inner))**2

            x = get_stereo_proj_torch(z)
            h = self(x)
            euclidean_sq = ((h.unsqueeze(1) - h.unsqueeze(0))**2).sum(2)
            if torch.isnan(euclidean_sq).any():
                print(f"NaN at epoch {epoch}")
                break
            loss = torch.mean((arclengths_sq - euclidean_sq)**2)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            lossE.append(loss.item())
            
            pbar.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
            if np.mod(epoch, 1000) == 0 and epoch != 0:
                lossAvg.append(np.mean(lossE))
                lossE = []

        print(f'Done. Final loss: {lossE[-1]}')
