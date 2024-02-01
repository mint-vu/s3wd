import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time
import os
from scipy.stats import gaussian_kde
import numpy as np
from config import Config, parse_args
from model import MNISTAE, C10AE
from s3w import ri_s3wd, ri_s3wd_unif, ari_s3wd, ari_s3wd_unif, s3wd, s3wd_unif
from sw import swd
from w import g_wasserstein
from ssw import sswd_unif, sswd
from utils import rand_unif, fibonacci_sphere
from misc_utils import generate_rand_projs
from s3w_utils import RotationPool, hStar, epsilon_projection, get_stereo_proj_torch, unif_hypersphere
import torch.nn.functional as F

def main():
    args = parse_args()
    Config.loss1 = args.loss1
    Config.loss2 = args.loss2
    Config.d = args.d
    Config.dataset = args.dataset
    Config.prior = args.prior
    Config.device = args.gpus
    Config.lr = args.lr
    Config.n_epochs = args.epochs
    Config.beta = args.beta



    if Config.dataset == 'c10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        model = C10AE(embedding_dim=Config.d)
    elif Config.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        model = MNISTAE(embedding_dim=Config.d)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    model = model.to(Config.device)
    criterion1 = nn.BCELoss()
    criterion2 = get_loss_func(Config.loss2, Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    start_time = time.time()
    if args.type == 'ae':
        train_ae(model, train_loader, criterion1, criterion2, optimizer, Config.device)
    elif args.type == 'swae':
        train_swae(model, train_loader, criterion1, criterion2, Config.beta, optimizer, Config.device)
    total_time = time.time() - start_time
    time_per_epoch = total_time / Config.n_epochs

    embeddings, BCE_losses = get_embs(model, test_loader, Config.device)
    avg_BCE = torch.cat(BCE_losses).mean().item()

    test_W2 = []
    for embedding in embeddings:
        sphere_samples = rand_unif(embedding.size(0), embedding.size(1), Config.device)
        embedding = embedding.to(Config.device) 
        W2_dist = g_wasserstein(embedding, sphere_samples)
        test_W2.append(W2_dist)
    avg_test_W2 = torch.tensor(test_W2).mean().item()

    os.makedirs('results', exist_ok=True)
    result_line = (
        f"Dataset: {Config.dataset}, "
        f"Learning Rate: {Config.lr}, "
        f"Epochs: {Config.n_epochs}, "
        f"Embedding Dim: {Config.d}, "
        f"Prior: {Config.prior}, "
        f"Loss 1: {Config.loss1}, "
        f"Loss 2: {Config.loss2}, "
        f"Beta: {Config.beta}, "
        f"Total Time: {total_time:.4f}s, "
        f"Time per Epoch: {time_per_epoch:.4f}s, "
        f"Spherical Wasserstein Distance: {avg_test_W2:.4f}, "
        f"Average BCE: {avg_BCE:.4f}\n"
    )
    with open('results/all_results.txt', 'a') as f:
        f.write(result_line)

def train_ae(model, train_loader, criterion1, criterion2, optimizer, device):
    for epoch in tqdm(range(Config.n_epochs), desc='Training AE'):
        for data in train_loader:
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs, embeddings = model(images)
            loss1 = criterion1(outputs, images)
            batch_prior_samples = get_prior(Config.prior, Config.d, images.size(0), device)
            loss2 = criterion2(embeddings, batch_prior_samples)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
    save_filename = f"results/AE_{Config.dataset}_lr{Config.lr}_epoch{Config.n_epochs}_dim{Config.d}_prior{Config.prior}_loss1{Config.loss1}_loss2{Config.loss2}.pt"
    torch.save(model.state_dict(), save_filename)

def train_swae(model, train_loader, criterion1, criterion2, beta, optimizer, device):
    for epoch in tqdm(range(Config.n_epochs), desc='Training SW'):
        for data in train_loader:
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs, embeddings = model(images)
            loss1 = criterion1(outputs, images)
            batch_prior_samples = get_prior(Config.prior, Config.d, images.size(0), device)
            loss2 = criterion2(embeddings, batch_prior_samples)
            loss = loss1 + beta * loss2
            loss.backward()
            optimizer.step()
    save_filename = f"results/SWAE_{Config.dataset}_lr{Config.lr}_epoch{Config.n_epochs}_dim{Config.d}_prior{Config.prior}_loss1{Config.loss1}_loss2{Config.loss2}_beta{Config.beta}.pt"
    torch.save(model.state_dict(), save_filename)

def get_loss_func(loss_name, device):
    if loss_name == 's3w':
        return lambda X, Y: s3wd(X, Y, p=2)
    elif loss_name.startswith('ri'):
        rotations = int(loss_name[2:])
        return lambda X, Y: ri_s3wd(X, Y, p=2, n_rotations=rotations)
    elif loss_name.startswith('ari'):
        rotations = int(loss_name[3:])
        return lambda X, Y: ari_s3wd(X, Y, p=2, n_rotations=rotations, pool_size=100)
    elif loss_name == 'ssw':
        return lambda X, Y: sswd(X, Y, num_projections=100, p=2, device=device)
    elif loss_name == 'sw':
        return lambda X, Y: swd(X, Y, n_projs=100,p=2,device=device)
    elif loss_name == 'mse':
        return lambda X, Y: nn.MSELoss()(X,Y)

def get_prior(prior, dim, n_samples, device):
    if prior == 'uniform':
        return rand_unif(n_samples, dim, device)
    elif prior == 'vmf':
        vmf_samples = fibonacci_sphere(n_samples)
        return torch.tensor(vmf_samples, dtype=torch.float, device=device)


def get_embs(model, data_loader, device):
    model.eval()
    embeddings = []
    BCE_losses = []
    with torch.no_grad():
        for data in data_loader:
            images, _ = data
            images = images.to(device)
            outputs, embedding = model(images)
            images = images.clamp(0, 1)
            outputs = outputs.clamp(0, 1)
            BCE_loss = F.binary_cross_entropy(outputs, images, reduction='none')
            BCE_loss = BCE_loss.mean(dim=[1, 2, 3]).detach().cpu()
            BCE_losses.append(BCE_loss)
            embeddings.append(embedding.detach().cpu())
    return embeddings, BCE_losses

if __name__ == '__main__':
    main()
