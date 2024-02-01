import argparse

class Config:
    type = 'ae'
    batch_size = 512
    lr = 1e-4
    epochs = 100
    d = 11
    beta = 1
    device = 'cuda:0'  
    loss1 = 'BCE'
    loss2 = 'MSE'
    dataset = 'CIFAR10'
    prior = 'uniform'


def parse_args():
    parser = argparse.ArgumentParser(description='training configs')
    parser.add_argument('--type', type=str, default='ae', choices=['ae', 'swae'], help='which ae?')
    parser.add_argument('--loss1', type=str, help='loss1')
    parser.add_argument('--loss2', type=str, help='loss2')
    parser.add_argument('--beta', type=float, help='beta')
    parser.add_argument('--d', type=int, help='embedding dim')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'c10'], help='dataset')
    parser.add_argument('--prior', type=str, choices=['uniform', 'vmf'], help='prior')
    parser.add_argument('--gpus', type=str, help='device')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    
    args = parser.parse_args()
    return args