import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

from itertools import cycle
from copy import deepcopy

from tqdm.auto import tqdm
from IPython.display import display, clear_output

from utils.nf.exp_map import create_NF
from utils.vi import *
from utils.misc import *

"""
Experiment 2, section H.7.3
Refactored from Bonet et al. 2023 (https://github.com/clbonet/spherical_sliced-wasserstein)
"""

def run_exp(n_epochs, V, d_func, d_args, lr=1e-3, model=None, d=3, n_particles=100, steps_mcmc=20, 
             dt_mcmc=1e-3, device='cpu', snapshot_t=None):
    snapshots = []
    if model is None:
        model = make_NF(d, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = tqdm(total=n_epochs, desc="Training")
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
        pbar.update(1)
    pbar.close()
    return model, snapshots if snapshot_t else None


def main():
    args = parser.parse_args()
    V = lambda x: -torch.log(target_density(x))
    model, snapshots = run_exp(10001, V, args.lr, plot=False, n_particles=500, dt_mcmc=1e-1)
    L_kl = np.zeros((args.ntry, 10001//100 +2))
    L_ess = np.zeros((args.ntry, 10001//100 +2))
    for k in range(args.ntry):
        for i, (_, snapshot_model) in enumerate(snapshots):
            z = torch.randn((500, 3), device=device)
            z = F.normalize(z, p=2, dim=-1)
            x, log_det = snapshot_model(z)
            log_prob = np.log(1 / (4 * np.pi)) * np.ones(z.shape[0]) - log_det.detach().cpu().numpy()
            _, kl, ess = kl_ess(log_prob, s2_target(x[-1].detach().cpu().numpy()))
            L_kl[k, i] = kl
            L_ess[k, i] = ess/z.shape[0] * 100
    kl_filename = f"./kl_{args.distance}.csv"
    ess_filename = f"./ess_{args.distance}.csv"
    np.savetxt(kl_filename, L_kl, delimiter=",")
    np.savetxt(ess_filename, L_ess, delimiter=",")

if __name__ == "__main__":
    main()