"""Code to load the MNIST, CIFAR-10, and ImageNet datasets."""

import torch
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import yaml

config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)
imagenet_path = parsed_config['imagenet_path']
tiny_imagenet_path = parsed_config['tiny_imagenet_path']

def dataset_loader(dataset: str, batch_size: int, 
    train_set_fraction: float, validate: bool = True,
    distributed: bool = False
    ) -> tuple([DataLoader, DataLoader]):
    """Load a specified dataset.

    Args:
        dataset: String specifying which dataset to load.
        batch_size: Batch size.
        train_set_fraction: Fraction of training set to use.
        validate: Whether or not to hold out part of the training set for validation.
        distributed: Whether or not loading will be parallelized.

    Returns:
        Training and validation set data loaders.

    """
    if dataset == 'mnist':
        return mnist_loader(
            batch_size=batch_size, train_set_fraction=train_set_fraction, 
            validate=validate, distributed=distributed)
    elif dataset == 'cifar':
        return cifar_loader(
            batch_size=batch_size, train_set_fraction=train_set_fraction,
            validate=validate, distributed=distributed)
    elif dataset == 'cifar100':
        return cifar_loader(
            batch_size=batch_size, train_set_fraction=train_set_fraction,
            validate=validate, distributed=distributed,cifar10 = False)
    elif dataset == 'tiny_imagenet':
        if not validate:
            raise ValueError('Test set not available for TinyImageNet.')
        return tiny_imagenet_loader(
            batch_size=batch_size, train_set_fraction=train_set_fraction,
            distributed=distributed)
    else:
        if not validate:
            raise ValueError('Test set not available for ImageNet.')
        return imagenet_loader(
            batch_size=batch_size, train_set_fraction=train_set_fraction,
            distributed=distributed)

def imagenet_loader(
    batch_size: int, train_set_fraction: float = 1.0, workers: int = 10, 
    distributed: bool = False
    ) -> tuple([DataLoader, DataLoader]):
    """Loader for the ImageNet dataset.

    Args:
        batch_size: Batch size.
        workers: Number of workers.
        distributed: Whether or not to parallelize.

    Returns:
        Training and validation set loaders.

    """
    traindir = os.path.join(imagenet_path, 'train')
    valdir = os.path.join(imagenet_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_set_size = int(1281167 * train_set_fraction)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_subset, _ = torch.utils.data.random_split(
            train_dataset, [train_set_size, 1281167-train_set_size])

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader

def tiny_imagenet_loader(
    batch_size: int, train_set_fraction: float = 1.0, workers: int = 10, 
    distributed: bool = False
    ) -> tuple([DataLoader, DataLoader]):
    """Loader for the ImageNet dataset.

    Args:
        batch_size: Batch size.
        workers: Number of workers.
        distributed: Whether or not to parallelize.

    Returns:
        Training and validation set loaders.

    """
    traindir = os.path.join(tiny_imagenet_path, 'train')
    valdir = os.path.join(tiny_imagenet_path, 'val')

    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_set_size = int(100000 * train_set_fraction)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_subset, _ = torch.utils.data.random_split(
            train_dataset, [train_set_size, 100000-train_set_size])

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader

class TransformedDataset(Dataset):
    """Wrapper class for augmented dataset.
    
    From https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580.

    Args:
        dataset (Dataset): Original dataset.
        transform (transforms.Compose): Data augmentation to apply.
    
    """

    def __init__(self, dataset: Dataset, 
        transform: transforms.Compose):
        """Instantiate object.
        
        Attributes:
            dataset: Original dataset.
            transform: Data augmentation to apply.

        """
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int):
        """Get augmented image."""
        return self.transform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


def cifar_loader(batch_size: int, 
    train_set_fraction: float = 1.0, validate: bool = True, 
    distributed: bool = False, cifar10 : bool = True
    ) -> tuple([DataLoader, DataLoader]):
    """Load the CIFAR-10 training set and split into training and validation.

    Based on https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    
    Args:
        batch_size: Batch size.
        device: Device being used.
        train_set_fraction: Fraction of training set to use.
        validate: Whether or not to hold out part of the training set for validation.
        distributed: Whether or not we are conducting parallel computation.

    Returns:
        Training and validation set loaders.

    """
    if cifar10:
        train_set = datasets.CIFAR10(root='./data', train=True,
            download=True)
    else:
        train_set = datasets.CIFAR100(root='./data', train=True,
            download=True)
    train_set_size = int(train_set_fraction * 50000)

    train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    if validate:
        #Split original training set into new training and validation sets
        if train_set_size == 50000:
            raise ValueError('Cannot use the full training set if performing validation.')
        train_subset, val_subset = torch.utils.data.random_split(
            train_set, [train_set_size, 50000-train_set_size])
        train_transformed = TransformedDataset(train_subset, train_transforms)
        val_transformed = TransformedDataset(val_subset, test_transforms)
        test_loader = DataLoader(
            val_transformed, batch_size=batch_size, shuffle=False)
    else:
        train_subset, _ = torch.utils.data.random_split(
            train_set, [train_set_size, 50000-train_set_size])
        train_transformed = TransformedDataset(train_subset, train_transforms)
        if cifar10:
            test_set = datasets.CIFAR10(root='./data', train=False,
                download=True, transform=test_transforms)
        else:
            test_set = datasets.CIFAR100(root='./data', train=False,
                download=True, transform=test_transforms)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if distributed:
        sampler = DistributedSampler(train_transformed)
    else:
        sampler = None

    train_loader = DataLoader(
        train_transformed, batch_size=batch_size, sampler=sampler, num_workers=0,
        pin_memory=True, shuffle=(not distributed))

    return train_loader, test_loader

def mnist_loader(batch_size: int, train_set_fraction: float,
    validate: bool = True, distributed: bool = False
    ) -> tuple([DataLoader, DataLoader]):
    """Load the MNIST training set and split into training and validation.

    Code based on PyTorch example code
    https://github.com/pytorch/examples/blob/master/mnist/main.py

    Args:
        batch_size: Batch size.
        train_set_fraction: Fraction of training set to be used.
        validate: Whether or not to hold out a validation set.
        distributed: Whether or not we are conducting parallel computation.

    Returns:
        Training set and validation set loaders.

    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    #Extract MNIST training set
    train_set = datasets.MNIST('../data', train=True, download=True,
        transform=transform)
    train_set_size = int(train_set_fraction * 60000)

    if validate:
        #Split original training set into new training and validation sets
        if train_set_size == 60000:
            raise ValueError('Cannot use the full training set if performing validation.')
        train_subset, valid_subset = torch.utils.data.random_split(train_set,
            [train_set_size, 60000 - train_set_fraction])
        test_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    else:
        train_subset, _ = torch.utils.data.random_split(
            train_set, [train_set_size, 60000-train_set_size])
        test_set = datasets.MNIST('../data', train=False, download=True,
        transform=transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if distributed:
        sampler = DistributedSampler(train_subset)
    else:
        sampler = None
        
    train_loader = DataLoader(
        train_subset, sampler=sampler, batch_size=batch_size,
        num_workers=1, shuffle=True)

    return train_loader, test_loader