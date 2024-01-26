import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import trange

# from utils import plot, s3w, vmf, misc


def run_exp(X_target, X0, d_func, d_args, device, n_steps=2000, lr=0.1, batch_size=500):
    """
    Performs gradient flow on particles using a specified distance function.

    Args:
    - X_target (torch.Tensor): Target distribution tensor.
    - distance_fn (function): Distance function to be used.
    - distance_fn_args (dict): Arguments required for the distance function.
    - device (torch.device): Device to perform computations on.
    - n_steps (int): Number of gradient steps.
    - lr (float): Learning rate.
    - batch_size (int): Batch size for processing.

    Returns:
    - List of tensors representing the state of particles at each step.
    - List of loss values.
    """

    loader = DataLoader(X_target, batch_size=batch_size, shuffle=True)
    dataiter = cycle(loader)

    X0 = X0.to(device)
    X0.requires_grad_(True)

    L = [X0.clone()]
    L_loss = []

    pbar = trange(n_steps)

    for k in pbar:
        X_batch = next(dataiter).type(torch.float).to(device)

        distance = d_func(X_batch, X0, **d_args)
        grad_x0 = torch.autograd.grad(distance, X0)[0]

        X0 = X0 - lr * grad_x0
        X0 = F.normalize(X0, p=2, dim=1)

        if torch.any(torch.isnan(grad_x0)):
            pass

        L_loss.append(distance.item())
        L.append(X0.clone().detach())
        pbar.set_postfix_str(f"Loss = {distance.item():.3f}")

    return L, L_loss