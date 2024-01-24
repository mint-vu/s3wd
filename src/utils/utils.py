import torch
import numpy as np


def is_tensor(x):
    return torch.is_tensor(x)

def generate_rand_projs(dim, n_projs=1000):
    projs = torch.randn(n_projs, dim).to(torch.float64)
    return projs / torch.norm(projs, p=2, dim=1, keepdim=True)

def rand_u_hypersphere(N, p):
    if is_tensor(N):
        v = torch.randn(N, p)
        v /= torch.norm(v, dim=1, keepdim=True)
    else:
        v = np.random.normal(0, 1, (N, p))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def rand_t_marginal(kappa, p, N=1):
    if is_tensor(kappa):
        b = (p - 1.0) / (2.0 * kappa + torch.sqrt(4.0 * kappa**2 + (p - 1.0)**2))
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa * x0 + (p - 1.0) * torch.log(1.0 - x0**2)
        samples = torch.zeros(N, 1)
        for i in range(N):
            while True:
                Z = torch.distributions.beta.Beta((p - 1.0) / 2.0, (p - 1.0) / 2.0).sample()
                U = torch.rand(1)
                W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)
                if kappa * W + (p - 1.0) * torch.log(1.0 - x0 * W) - c >= torch.log(U):
                    samples[i] = W
                    break
    else:
        b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa**2 + (p - 1.0)**2))
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0**2)
        samples = np.zeros((N, 1))
        for i in range(N):
            while True:
                Z = np.random.beta((p - 1.0) / 2.0, (p - 1.0) / 2.0)
                U = np.random.uniform(0, 1)
                W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)
                if kappa * W + (p - 1.0) * np.log(1.0 - x0 * W) - c >= np.log(U):
                    samples[i] = W
                    break
    return samples

def spherical_to_euclidean(sph_coords):
    '''
        Source: https://github.com/katalinic/sdflows/blob/0f319d8ae6e2c858061a0a31880d4b70f69b6a64/utils.py#L4
    '''
    if sph_coords.ndim == 1:
        sph_coords = np.expand_dims(sph_coords, 0)
        
    theta, phi = np.split(sph_coords, 2, 1)
    return np.concatenate((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ), 1)

def euclidean_to_spherical(euc_coords):
    '''
        Source: https://github.com/katalinic/sdflows/blob/0f319d8ae6e2c858061a0a31880d4b70f69b6a64/utils.py#L15
    '''
    if euc_coords.ndim == 1:
        euc_coords = np.expand_dims(euc_coords, 0)
    x, y, z = np.split(euc_coords, 3, 1)
    return np.concatenate((
        np.pi + np.arctan2(-y, -x),
        np.arccos(z)
    ), 1)

def spherical_to_euclidean_torch(x):
    '''
        Source: https://github.com/clbonet/Spherical_Sliced-Wasserstein/blob/main/lib/utils_sphere.py
    '''
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    theta = x[:,0]
    phi = x[:,1]
    
    xx = torch.sin(phi)*torch.cos(theta)
    yy = torch.sin(phi)*torch.sin(theta)
    zz = torch.cos(phi)
    
    return torch.cat([xx[:,None],yy[:,None],zz[:,None]], dim=-1)

