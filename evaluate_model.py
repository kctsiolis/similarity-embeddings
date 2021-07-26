"""Train a model ("teacher") in supervised fashion on an image dataset.

Models supported:
    ResNet18 (imported from PyTorch)
    ResNet50 (imported from PyTorch)
    ResNet152 (imported from PyTorch)

Datasets supported:
    MNIST
    CIFAR-10
    ImageNet

"""

import argparse
import torch
from torch import nn
import numpy as np
from models import get_model
from mnist import mnist_train_loader
from cifar import cifar_train_loader
from imagenet import imagenet_train_loader
from training import predict
from logger import Logger

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='Batch size (default: 64)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Name of CUDA device being used (if any).')
    parser.add_argument('--augs', type=str, default='all', choices=['normalize', 'flip', 'all'],
                        help='Data augmentations to use on training set (normalize only, normalize' \
                            'and flip, normalize, flip, and crop).')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet152',
                        'resnet50_pretrained', 'simclr_pretrained'],
                        help='Choice of model.')
    parser.add_argument('--load-path', type=str,
                        help='Path to the teacher model.')
    parser.add_argument('--all-subsets', action='store_true',
                        help='Evaluate on all subsets of the data (including the training set). Otherwise ' \
                            'only evaluate on validation set.')
    args = parser.parse_args()

    return args    

def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Training the teacher model')
    args = get_args(parser)

    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    #Get the data
    if args.dataset == 'mnist':
        train_loader, valid_loader = mnist_train_loader(batch_size=args.batch_size,
            device=device)
    elif args.dataset == 'cifar':
        train_loader, valid_loader = cifar_train_loader(batch_size=args.batch_size,
            device=device, augs=args.augs)
    else:
        train_loader, valid_loader = imagenet_train_loader(batch_size=args.batch_size)

    logger = Logger('teacher', args.dataset, args, save=False)
            
    one_channel = args.dataset == 'mnist'
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    model = get_model(args.model, load=True, load_path=args.load_path, 
        one_channel=one_channel, num_classes=num_classes)
    model.to(device)

    if args.all_subsets:
        predict(model, device, train_loader, nn.CrossEntropyLoss(), logger, 'Training')
    predict(model, device, valid_loader, nn.CrossEntropyLoss(), logger, 'Validation')

if __name__ == '__main__':
    main()