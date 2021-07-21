"""Code for loading the MNIST dataset.

Code based on PyTorch example code
https://github.com/pytorch/examples/blob/master/mnist/main.py

"""

import torch
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

def mnist_train_loader(batch_size: int, device: torch.device,
    distributed: bool = False) -> tuple([torch.utils.data.DataLoader, 
    torch.utils.data.DataLoader]):
    """Load the MNIST training set and split into training and validation.

    Args:
        batch_size: Batch size.
        device: Device being used.
        distributed: Whether or not we are conducting parallel computation.

    Returns:
        Training set and validation set loaders.

    """
    train_kwargs = {'batch_size': batch_size}
    valid_kwargs = {'batch_size': batch_size}
    if device.type == 'cuda':
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': not distributed
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

    if distributed:
        sampler = DistributedSampler(train_subset)
    else:
        sampler = None
        
    train_loader = torch.utils.data.DataLoader(train_subset, sampler=sampler, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_subset, **valid_kwargs)

    return train_loader, valid_loader