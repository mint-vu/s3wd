import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle
from tqdm import trange

def run_exp(X_target, X0, model, d_func, d_args, device, n_steps=2000, lr=1e-3, batch_size=500, snapshot_t=None):
    """
    Runs experiments using a Normalizing Flow model.

    Args:
    - X_target (torch.Tensor): Target distribution tensor.
    - X0 (torch.Tensor): Initial state of particles.
    - model (torch.nn.Module): Normalizing Flow model.
    - optimizer (torch.optim.Optimizer): Optimizer for the model.
    - d_func (function): Distance function.
    - d_args (dict): Arguments for the distance function.
    - device (torch.device): Device.
    - n_steps (int): Number of gradient steps.
    - lr (float): Learning rate.
    - batch_size (int): Batch size for processing.

    Returns:
    - List of tensors representing the state of particles at each step.
    - List of loss values.
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_target)
    dataiter = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    X0 = X0.to(device)
    X0.requires_grad_(True)

    L = [X0.clone()]
    L_loss = []
    pbar = trange(n_steps)
    
    saved_snapshots = {}

    for k in pbar:
        X_target_batch = next(dataiter)[0].type(torch.float).to(device)
        
        optimizer.zero_grad()
        
        z, _ = model(X0)

        distance = d_func(z[-1], X_target_batch, **d_args)
        distance.backward()

        optimizer.step()

        with torch.no_grad():
            X0 = F.normalize(X0, p=2, dim=1)

        L_loss.append(distance.item())
        L.append(X0.clone().detach())
        pbar.set_postfix_str(f"Loss = {distance.item():.3f}")
        
        if snapshot_t is not None:
            if k in snapshot_t:
                noise = F.normalize(torch.randn((1000, 3), device=device), p=2, dim=-1)
                z0, _ = model(noise)
                saved_snapshots[k] = z0[-1].detach().cpu().numpy()

    if snapshot_t: return L, L_loss, saved_snapshots
    return L, L_loss