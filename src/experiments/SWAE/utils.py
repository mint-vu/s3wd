import torch
import numpy as np

def rand_unif(N, d, device):
    samples = torch.randn(N, d, device=device)
    samples_ = samples / samples.norm(dim=1, keepdim=True)
    return samples_

def fibonacci_sphere(samples=1000):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    y = 1 - (2 * np.arange(samples) / (samples - 1))
    radius = np.sqrt(1 - y*y)
    theta = phi * np.arange(samples)
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    points = np.stack([x, y, z], axis=-1)
    return points
