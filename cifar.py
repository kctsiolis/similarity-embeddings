"""Code for loading the CIFAR-10 dataset.

Based on https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

"""

import torch
from torchvision import datasets, transforms

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


def cifar_train_loader(train_batch_size: int = 64, valid_batch_size: int = 1000, 
    device: str = 'cpu', augs: str = 'normalize'
    ) -> tuple([torch.utils.data.DataLoader, torch.utils.data.DataLoader]):
    """Load the CIFAR-10 training set and split into training and validation.
    
    Args:
        train_batch_size: Training set batch size.
        valid_batch_size: Validation set batch size.
        device: String indicating which device to use.
        augs: Data augmentations to use.

    Returns:
        Training and validation set loaders.

    """
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

    train_transforms = [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    valid_transforms = [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

    if augs == 'all':
        train_transforms.insert(0, transforms.RandomHorizontalFlip())
        train_transforms.insert(0, transforms.RandomCrop(32, padding=4))
    elif augs == 'flip':
        train_transforms.insert(0, transforms.RandomHorizontalFlip())

    transform_train = transforms.Compose(train_transforms)

    transform_valid = transforms.Compose(valid_transforms)

    train_set = datasets.CIFAR10(root='./data', train=True,
        download=True)

    #Split original training set into new training and validation sets
    train_subset, valid_subset = torch.utils.data.random_split(train_set,
        [40000, 10000])

    train_transformed = TransformedDataset(train_subset, transform_train)
    valid_transformed = TransformedDataset(valid_subset, transform_valid)

    train_loader = torch.utils.data.DataLoader(train_transformed,**train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_transformed, **valid_kwargs)

    return train_loader, valid_loader