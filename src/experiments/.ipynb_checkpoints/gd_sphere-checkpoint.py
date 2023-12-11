import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import trange

def run_exp(X_target, X0, d_func, d_args, device, n_steps=1001, lr=100, batch_size=500):
    """
    Performs gradient descent on the sphere using a specified distance function and initial particle state.

    Args:
    - X_target_loader (DataLoader): DataLoader for the target distribution.
    - X0 (torch.Tensor): Initial state of particles.
    - distance_fn (function): Distance function to be used.
    - distance_fn_args (dict): Arguments required for the distance function.
    - device (torch.device): Device to perform computations on.
    - n_steps (int): Number of gradient steps.
    - lr (float): Learning rate.

    Returns:
    - List of tensors representing the state of particles at each step.
    - List of loss values.
    """
#     X_target_loader = DataLoader(X_target, batch_size=batch_size, shuffle=True)
    
#     X0 = X0.to(device)
#     X0.requires_grad_(True)

#     L = [X0.clone()]
#     L_loss = []

#     pbar = trange(n_steps)

#     for k in pbar:
#         X_target = next(iter(X_target_loader)).type(torch.float).to(device)
        
#         sw = d_func(X_target, X0, **d_args)
#         grad_x0 = torch.autograd.grad(sw, X0)[0]
#         grad_S = grad_x0 - torch.sum(X0 * grad_x0, dim=-1)[:, None] * X0
#         v = -lr * grad_S
#         norm_v = torch.linalg.norm(v, axis=-1)[:, None]
        
#         X0 = X0 * torch.cos(norm_v) + torch.sin(norm_v) * v / norm_v
        
#         L_loss.append(sw.item())
#         L.append(X0.clone().detach())
#         pbar.set_postfix_str(f"loss = {sw.item():.3f}")


    X_target_loader = DataLoader(X_target, batch_size=batch_size, shuffle=True)
    data_iter = cycle(X_target_loader)

    X0 = X0.to(device)
    X0.requires_grad_(True)

    L = [X0.clone()]
    L_loss = []

    pbar = trange(n_steps)
    
    saved_snapshots = {}

    for k in pbar:
        X_target = next(data_iter).type(torch.float).to(device)
        
        sw = d_func(X_target, X0, **d_args)
        grad_x0 = torch.autograd.grad(sw, X0)[0]
        grad_S = grad_x0 - torch.sum(X0 * grad_x0, dim=-1)[:, None] * X0
        v = -lr * grad_S
        norm_v = torch.linalg.norm(v, axis=-1)[:, None]
        
        X0 = X0 * torch.cos(norm_v) + torch.sin(norm_v) * v / norm_v
        X0 = F.normalize(X0, p=2, dim=-1)  

        L_loss.append(sw.item())
        L.append(X0.clone().detach())
        pbar.set_postfix_str(f"loss = {sw.item():.3f}")
        

    return L, L_loss