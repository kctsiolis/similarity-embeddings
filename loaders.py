"""Code to load the MNIST, CIFAR-10, and ImageNet datasets."""

import torch
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

import h5py
import io
import json
import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import yaml

try:
    config = open('config.yaml', 'r')
except FileNotFoundError:
    config = open('../config.yaml', 'r')

parsed_config = yaml.load(config, Loader=yaml.FullLoader)
h5_path = parsed_config['imagenet_h5_path']
data_path = parsed_config['imagenet_path']

def dataset_loader(dataset: str, batch_size: int, 
    device: torch.device, distributed: bool = False):
    if dataset == 'mnist':
        return mnist_loader(batch_size=batch_size,
            device=device, distributed=distributed)
    elif dataset == 'cifar':
        return cifar_loader(batch_size=batch_size,
            device=device, distributed=distributed)
    else:
        return imagenet_loader(batch_size=batch_size,
            distributed=distributed)

def imagenet_loader(batch_size, workers=10, distributed=False):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader

def imagenet_h5_loader(batch_size, classes=[], shuffle=True,
                        workers=10, distributed=False):
    """Adapted from original code by Vikram Voleti. Used with his permission."""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])

    val_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])

    train_ds = ImageNetDataset('train', train_transform, classes)
    val_ds = ImageNetDataset('val', val_transform, classes)

    if distributed:
        sampler = DistributedSampler(train_ds)
        shuffle = False
    else:
        sampler = None

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, sampler=sampler, drop_last=True)

    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)

    return train_dl, val_dl


class ImageNetDataset(Dataset):
    def __init__(self, split, transform=transforms.Compose([transforms.ToTensor()]), classes=[]):
        self.h5_path = h5_path     # Path to ilsvrc2012.hdf5
        self.split = split
        self.transform = transform
        self.classes = classes

        assert os.path.exists(self.h5_path), f"ImageNet h5 file path does not exist! Given: {self.h5_path}"
        assert self.split in ["train", "val", "test"], f"split must be 'train' or 'val' or 'test'! Given: {self.split}"

        self.N_TRAIN = 1281167
        self.N_VAL = 50000
        self.N_TEST = 100000

        if self.split in ['train', 'val']:
            if len(self.classes) > 0:
                class_idxs_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet.json")))[self.split]
                self.class_idxs = sorted([i for c in self.classes for i in class_idxs_dict[str(c)]])
                del class_idxs_dict
                self.n = len(self.class_idxs)
            else:
                if self.split == 'train':
                    self.n  = self.N_TRAIN
                elif self.split == 'val':
                    self.n = self.N_VAL
        else:
            self.n = self.N_TEST

        self.h5_data = None

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        # Get class idx
        if len(self.classes) > 0 and self.split in ['train', 'val']:
            idx = self.class_idxs[idx]

        # Correct idx
        if self.split == 'val':
            idx += self.N_TRAIN
        elif self.split == 'test':
            idx += self.N_TRAIN + self.N_VAL

        # Read h5 file
        if self.h5_data is None:
            self.h5_data = h5py.File(self.h5_path, mode='r')

        # Extract info
        image = self.transform(Image.open(io.BytesIO(self.h5_data['encoded_images'][idx])).convert('RGB'))
        target = torch.from_numpy(self.h5_data['targets'][idx])[0].long() if self.split != 'test' else None

        return image, target

class TransformedDataset(torch.utils.data.Dataset):
    """Wrapper class for augmented dataset.
    
    From https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580.

    Args:
        dataset (torch.utils.data.Dataset): Original dataset.
        transform (transforms.Compose): Data augmentation to apply.
    
    """

    def __init__(self, dataset: torch.utils.data.Dataset, 
        transform: transforms.Compose):
        """Instantiate object.
        
        Args:
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


def cifar_loader(batch_size: int, device: torch.device, 
    distributed: bool = False,
    ) -> tuple([torch.utils.data.DataLoader, torch.utils.data.DataLoader]):
    """Load the CIFAR-10 training set and split into training and validation.

    Based on https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    
    Args:
        batch_size: Batch size.
        device: Device being used.
        distributed: Whether or not we are conducting parallel computation.

    Returns:
        Training and validation set loaders.

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

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    train_set = datasets.CIFAR10(root='./data', train=True,
        download=True)

    #Split original training set into new training and validation sets
    train_subset, valid_subset = torch.utils.data.random_split(train_set,
        [40000, 10000])

    train_transformed = TransformedDataset(train_subset, train_transforms)
    valid_transformed = TransformedDataset(valid_subset, valid_transforms)

    if distributed:
        sampler = DistributedSampler(train_transformed)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(train_transformed, sampler=sampler, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_transformed, **valid_kwargs)

    return train_loader, valid_loader

def mnist_loader(batch_size: int, device: torch.device,
    distributed: bool = False) -> tuple([torch.utils.data.DataLoader, 
    torch.utils.data.DataLoader]):
    """Load the MNIST training set and split into training and validation.

    Code based on PyTorch example code
    https://github.com/pytorch/examples/blob/master/mnist/main.py

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