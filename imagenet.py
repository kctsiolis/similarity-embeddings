"""Code to load the ImageNet dataset.

Adapted from original code by Vikram Voleti. Used with his permission.

"""

import h5py
import io
import json
import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import yaml

try:
    config = open('../config.yaml', 'r')
except FileNotFoundError:
    config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)
h5_path = parsed_config['imagenet_path']

# classes : https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57

def imagenet_train_loader(batch_size, classes=[], shuffle=True,
                        workers=10, distributed=False):

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