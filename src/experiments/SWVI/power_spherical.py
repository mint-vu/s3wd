import torch
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
from utils.power_spherical_utils import PowerSpherical
from utils.vi_utils import ULA_sphere

from itertools import cycle
from tqdm.auto import tqdm
from IPython.display import display, clear_output

def run_exp(n_epochs, V, d_func, d_args, device, lr=1e-3, d=3, n_particles=100, 
                         steps_mcmc=20, dt_mcmc=1e-3, snapshot_t=None):
    """
    Experiment 1, section H.7.1
    Refactored from Bonet et al. 2023 (https://github.com/clbonet/spherical_sliced-wasserstein)

    Args:
    - n_epochs: number of epochs
    - V: potential function
    - d_func: distance function
    - d_args: arguments for distance function
    - device: device
    - lr: learning rate
    - d: dimension
    - n_particles: number of particles
    - steps_mcmc: number of steps for MCMC
    - dt_mcmc: step size for MCMC
    - snapshot_t (list of int, optional): timesteps to take snapshots

    Returns:
    - Tuple containing final states of m and kappa, list of m over time, list of kappa over time, and snapshots if snapshot_t is provided
    """
    L_m = []
    L_kappa = []
    snapshots = []

    kappa = torch.tensor(0.1, device=device, requires_grad=True)
    m = F.normalize(torch.ones(d, device=device), p=2, dim=-1).requires_grad_(True)

    pbar = tqdm(total=n_epochs, desc="Training")
    display(pbar)  

    for e in range(n_epochs):
        ps = PowerSpherical(m, kappa)
        z0 = ps.rsample((n_particles,))
        zt, _ = ULA_sphere(V, device=device, init_particles=z0, d=d,
                           n_steps=steps_mcmc, n_particles=n_particles, dt=dt_mcmc)

        distance = d_func(z0, zt, **d_args)
        distance.backward()

        with torch.no_grad():
            if m.requires_grad:
                m -= lr * m.grad
                m = F.normalize(m, p=2, dim=-1)
                m.requires_grad_(True)

            if kappa.requires_grad:
                kappa -= 100 * lr * kappa.grad
                kappa.requires_grad_(True)

        if m.grad is not None:
            m.grad.zero_()
        if kappa.grad is not None:
            kappa.grad.zero_()

        if snapshot_t and e in snapshot_t:
            snapshots.append((e, m.detach().clone(), kappa.detach().clone()))

        L_m.append(m.detach().cpu().numpy())
        L_kappa.append(kappa.item())

        pbar.set_postfix(Loss=distance.item())
        pbar.update(1)  


        clear_output(wait=True)
        display(pbar)

    pbar.close()  

    return m.detach(), kappa.detach(), L_m, L_kappa, snapshots if snapshot_t else None