import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

from itertools import cycle
from copy import deepcopy

from tqdm.auto import tqdm
from IPython.display import display, clear_output

from utils.nf.normalizing_flows import make_NF
from utils.vi_utils import ULA_sphere

def swsvi_nf(n_epochs, V, d_func, d_args, lr=1e-3, model=None, d=3, n_particles=100, steps_mcmc=20, 
             dt_mcmc=1e-3, device='cpu', snapshot_t=None):
    """
    Runs SWVI with a Normalizing Flow model and a custom distance function.

    Args:
    - n_epochs (int): Number of epochs.
    - V (callable): Potential function.
    - d_func (function): Distance function to be used.
    - d_args (dict): Arguments required for the distance function.
    - lr (float): Learning rate.
    - model (torch.nn.Module, optional): Normalizing Flow model.
    - d (int): Dimension.
    - n_particles (int): Number of particles.
    - steps_mcmc (int): Number of steps for MCMC.
    - dt_mcmc (float): Step size for MCMC.
    - device (torch.device): Device to perform computations on.
    - snapshot_t (list of int, optional): Timesteps to take snapshots.

    Returns:
    - Final model and list of snapshots.
    """
    
    snapshots = []

    if model is None:
        model = make_NF(d, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(total=n_epochs, desc="Training Progress")
    display(pbar)  # Display the progress bar

    for e in range(n_epochs):
        optimizer.zero_grad()

        noise = F.normalize(torch.randn((n_particles, d), device=device), p=2, dim=-1)
        z0, _ = model(noise)

        zt, _ = ULA_sphere(V, device=device, init_particles=z0[-1], d=d, n_steps=steps_mcmc, n_particles=z0[-1].shape[0], dt=dt_mcmc)

        sw = d_func(z0[-1], zt, **d_args) / 2
        sw.backward()
        optimizer.step()

        if snapshot_t and e in snapshot_t:
            snapshots.append((e, deepcopy(model)))

        pbar.set_postfix(Loss=sw.item())
        pbar.update(1)

        clear_output(wait=True)
        display(pbar)

    pbar.close()

    return model, snapshots if snapshot_t else None