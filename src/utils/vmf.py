import torch
import numpy as np
from utils.misc import rand_t_marginal, rand_u_hypersphere
from scipy.linalg import null_space

def pdf_vmf(x, mu, kappa):
    return torch.exp(kappa * torch.matmul(mu, x.T))[0]

def pdf_von_Mises(theta,mu,kappa):
    pdf = np.exp(kappa * np.cos(theta - mu)) / (2.0*np.pi*iv(0,kappa))
    return pdf

def rand_vmf(mu, kappa, N=1):
    if torch.is_tensor(mu): mu = mu.numpy()  
    p = len(mu)
    mu = np.reshape(mu, (p, 1)).astype(float)
    mu /= np.linalg.norm(mu, axis=0)
    samples = np.zeros((N, p))
    t = rand_t_marginal(kappa, p, N) 
    xi = rand_u_hypersphere(N, p - 1) 
    samples[:, [0]] = t
    samples[:, 1:] = np.sqrt(1 - t**2) * xi
    O = null_space(mu.T)
    R = np.concatenate((mu, O), axis=1)
    return np.dot(R, samples.T).T

def fibonacci_sphere(samples=10):
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
    y = 1 - 2*np.arange(samples)/float(samples-1)
    radius = np.sqrt(1-y**2)
    theta = phi*np.arange(samples)
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    points = np.stack([x,y,z],axis=-1)
    return points
