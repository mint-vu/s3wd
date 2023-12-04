import torch

def generate_rand_projs(dim, n_projs=1000):
    projs = torch.randn(n_projs, dim)
    return projs / torch.norm(projs, p=2, dim=1, keepdim=True)

def sample_weight(kappa, dim):
    kappa = torch.tensor(kappa, dtype=torch.float32)
    dim = torch.tensor(dim, dtype=torch.float32)
    b = dim / (torch.sqrt(torch.tensor(4.0) * kappa**2 + dim**2) + 2 * kappa)
    x = (1 - b) / (1 + b)
    c = kappa * x + dim * torch.log(1 - x**2)
    while True:
        z = torch.distributions.beta.Beta((dim - 1) / 2, (dim - 1) / 2).sample()
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = torch.rand(1)
        if kappa * w + dim * torch.log(1 - x * w) - c >= torch.log(u):
            return w

def get_orthonormal_basis(mu):
    if torch.allclose(mu, torch.tensor([1.0, 0.0, 0.0])):
        return torch.eye(3)
    else:
        v = torch.tensor([1.0, 0.0, 0.0]) - torch.dot(mu, torch.tensor([1.0, 0.0, 0.0])) * mu
        v = v / v.norm(p=2)
        w = torch.cross(mu, v)
        return torch.stack([mu, v, w])

def sample_vmf(mu, kappa, num_samples):
    mu = torch.tensor(mu, dtype=torch.float32)
    dim = mu.size(0)
    mu = mu / mu.norm(p=2)
    result = torch.zeros(num_samples, dim)
    for n in range(num_samples):
        w = sample_weight(kappa, dim)
        v = torch.randn(dim - 1)
        v = v / v.norm(p=2)
        result[n, :] = torch.cat([torch.tensor([w]), torch.sqrt(1 - w**2) * v])
    U = get_orthonormal_basis(mu)
    return result @ U.T