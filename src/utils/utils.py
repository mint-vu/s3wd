import torch
import numpy as np
from scipy.linalg import null_space

def generate_rand_projs(dim, n_projs=1000):
    projs = torch.randn(n_projs, dim).to(torch.float64)
    return projs / torch.norm(projs, p=2, dim=1, keepdim=True)

def rand_u_hypersphere(N, p):
    v = np.random.normal(0, 1, (N, p))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def rand_t_marginal(kappa, p, N=1):
    b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa**2 + (p - 1.0)**2))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0**2)
    samples = np.zeros((N, 1))
    for i in range(N):
        while True:
            Z = np.random.beta((p - 1.0) / 2.0, (p - 1.0) / 2.0)
            U = np.random.uniform(0.0, 1.0)
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)
            if kappa * W + (p - 1.0) * np.log(1.0 - x0 * W) - c >= np.log(U):
                samples[i] = W
                break
    return samples

def rand_vmf(mu, kappa, N=1):
    p = len(mu)
    mu = np.reshape(mu, (p, 1)).astype(float)  # Ensure mu is a float array
    mu /= np.linalg.norm(mu, axis=0)
    samples = np.zeros((N, p))
    t = rand_t_marginal(kappa, p, N)
    xi = rand_u_hypersphere(N, p - 1)
    samples[:, [0]] = t
    samples[:, 1:] = np.sqrt(1 - t**2) * xi
    O = null_space(mu.T)
    R = np.concatenate((mu, O), axis=1)
    return np.dot(R, samples.T).T