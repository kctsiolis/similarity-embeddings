"""Code to load the MNIST, CIFAR-10, and ImageNet datasets."""

from multiprocessing.sharedctypes import Value
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

def get_loader(args) -> tuple([DataLoader, DataLoader]):
    """Load a specified dataset.

    Args:
        args: The command line arguments

    Returns:
        Training and validation set data loaders.

    """
    if args.dataset == 'cifar10':
        return cifar10_loader(args)
    elif args.dataset == 'cifar100':
        return cifar100_loader(args)
    elif args.dataset == 'tiny_imagenet':
        return tiny_imagenet_loader(args)
    elif args.dataset == 'imagenet':        
        return imagenet_loader(args)
    else:
        raise ValueError(f'Dataset {args.dataset} is not available')

def imagenet_loader(args) -> tuple([DataLoader, DataLoader]):
    """Loader for the ImageNet dataset.

    Args:
        args: The command line arguments.

    Returns:
        Training and validation set loaders.

    """

    if args.train_subset_indices_path is not None:
        with open(args.train_subset_indices_path, 'r+') as file:
            lines = file.readlines()
            indices = [int(x.strip()) for x in lines]


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

    train_set_size = int(1281167 * args.train_set_fraction)

    train_subset, _ = torch.utils.data.random_split(
            train_dataset, [train_set_size, 1281167-train_set_size])

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=10, pin_memory=True)

    return train_loader, val_loader, 1000

def tiny_imagenet_loader(args) -> tuple([DataLoader, DataLoader]):
    """Loader for the ImageNet dataset.

    Args:
        args: The command line arguments.

    Returns:
        Training and validation set loaders.

    """
    
    if args.train_subset_indices_path is not None:
        with open(args.train_subset_indices_path, 'r+') as file:
            lines = file.readlines()
            indices = [int(x.strip()) for x in lines]

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

    train_set_size = int(100000 * args.train_set_fraction)

    train_subset, _ = torch.utils.data.random_split(
            train_dataset, [train_set_size, 100000-train_set_size])

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=10, pin_memory=True)

    return train_loader, val_loader, 200

class TransformedDataset(Dataset):
    """Wrapper class for augmented dataset.
    
    From https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580.

    Args:
        dataset (Dataset): Original dataset.
        transform (transforms.Compose): Data augmentation to apply.
    
    """

    def __init__(self, dataset: Dataset, transform: transforms.Compose):
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


def cifar10_loader(args) -> tuple([DataLoader, DataLoader]):
    """Load the CIFAR-10 training set and split into training and validation.

    Based on https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    
    Args:
        args: Command line arguments.
        cifar10: If true, load CIFAR-10. Otherwise, load CIFAR-100.

    Returns:
        Training and validation set loaders.

    """
    if args.train_subset_indices_path is not None:
        with open(args.train_subset_indices_path, 'r+') as file:
            lines = file.readlines()
            indices = [int(x.strip()) for x in lines]


    train_set = datasets.CIFAR10(root='./data', train=True,
        download=True)

    train_set_size = int(args.train_set_fraction * 50000)

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

    train_subset, _ = torch.utils.data.random_split(
        train_set, [train_set_size, 50000-train_set_size])
    train_transformed = TransformedDataset(train_subset, train_transforms)


    test_set = datasets.CIFAR10(root='./data', train=False,
        download=True, transform=test_transforms)
    num_classes = 10

    if args.train_subset_indices_path is not None:
        train_transformed = torch.utils.data.Subset(train_transformed, indices)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers = 8, pin_memory= True)
    train_loader = DataLoader(
        train_transformed, batch_size=args.batch_size, num_workers=8,
        pin_memory=True)

    return train_loader, test_loader, num_classes

def cifar100_loader(args) -> tuple([DataLoader, DataLoader]):
    """Load the CIFAR-100 training set and split into training and validation.

    Based on https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    
    Args:
        args: Command line arguments.        

    Returns:
        Training and validation set loaders.

    """
    if args.train_subset_indices_path is not None:
        with open(args.train_subset_indices_path, 'r+') as file:
            lines = file.readlines()
            indices = [int(x.strip()) for x in lines]
    
    train_set = datasets.CIFAR100(root='./data', train=True,
        download=True)
    train_set_size = int(args.train_set_fraction * 50000)

    train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

    test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

    train_subset, _ = torch.utils.data.random_split(
        train_set, [train_set_size, 50000-train_set_size])
    train_transformed = TransformedDataset(train_subset, train_transforms)

    test_set = datasets.CIFAR100(root='./data', train=False,
        download=True, transform=test_transforms)
    num_classes = 100

    if args.train_subset_indices_path is not None:
        train_transformed = torch.utils.data.Subset(train_transformed, indices)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers = 8, pin_memory= True)
    train_loader = DataLoader(
        train_transformed, batch_size=args.batch_size, num_workers=8,
        pin_memory=True)

    return train_loader, test_loader, num_classes    