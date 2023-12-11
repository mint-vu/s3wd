import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle
from tqdm import trange
import numpy as np
import torch.optim as optim

def run_exp(model, X_target, d_func, d_args, device, n_steps=2001, batch_size=500, snapshot_t=None):
    """
    Trains a Normalizing Flow model using a custom distance function and captures snapshots.

    Args:
    - model (torch.nn.Module): Normalizing Flow model.
    - X_target (torch.Tensor): Target distribution tensor.
    - d_func (function): Custom distance function.
    - d_args (dict): Arguments for the distance function.
    - device (torch.device): Device to perform computations on.
    - n_steps (int): Number of training steps.
    - batch_size (int): Batch size for processing.
    - snapshot_t (list of int, optional): Timesteps to capture snapshots.

    Returns:
    - List of loss values.
    - Dictionary of snapshots if snapshot_t is provided.
    """
    
    optimizer = optim.Adam(model.parameters(), lr=d_args.get("lr", 1e-3))
    dataset = TensorDataset(X_target)
    dataiter = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    L_loss = []
    snapshots = {} if snapshot_t else None
    pbar = trange(n_steps)

    for k in pbar:
        X_batch = next(dataiter)[0].to(device)
        optimizer.zero_grad()

        z, _ = model(X_batch)
        distance = d_func(z[-1], **d_args)
        distance.backward()
        optimizer.step()

        L_loss.append(distance.item())
        pbar.set_postfix_str(f"Loss = {distance.item():.3f}")

        if snapshot_t and k in snapshot_t:
            with torch.no_grad():
                noise = F.normalize(torch.randn((1000, 3), device=device), p=2, dim=-1)
                z0, log_det = model(noise)
                snapshots[k] = (z0[-1], log_det)

    return L_loss, snapshots if snapshot_t else L_loss
