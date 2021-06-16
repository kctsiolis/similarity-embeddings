#Collection of functions to handle the MNIST dataset
#Code based on PyTorch example code
#https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
from torchvision import datasets, transforms
from sklearn.manifold import TSNE

def mnist_train_loader(train_batch_size=64, valid_batch_size=1000, device='cpu'):
    train_kwargs = {'batch_size': train_batch_size}
    valid_kwargs = {'batch_size': valid_batch_size}
    if device != "cpu":
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        valid_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    #Extract MNIST training set
    train_set = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)

    #Split original training set into new training and validation sets
    train_subset, valid_subset = torch.utils.data.random_split(train_set,
        [50000, 10000])
        
    train_loader = torch.utils.data.DataLoader(train_subset,**train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_subset, **valid_kwargs)

    return train_loader, valid_loader