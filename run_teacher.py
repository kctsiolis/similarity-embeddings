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
import sys
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from models import ResNet18, ResNet50, ResNet152
from mnist import mnist_train_loader
from cifar import cifar_train_loader
from imagenet import imagenet_train_loader
from training import train_sup
from logger import Logger

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                        help='Choice of optimizer for training.')
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine'], default='plateau',
                        help='Choice of scheduler for training.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience used in Plateau scheduler.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--early-stop', type=int, default=10, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--device', type=str, nargs='+', default=['cpu'],
                        help='Name of CUDA device(s) being used (if any). Otherwise will use CPU. \
                            Can also specify multiple devices (separated by spaces) for multiprocessing.')
    parser.add_argument('--augs', type=str, default='all', choices=['normalize', 'flip', 'all'],
                        help='Data augmentations to use on training set (normalize only, normalize' \
                            'and flip, normalize, flip, and crop).')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet152'],
                        help='Choice of model.')
    args = parser.parse_args()

    return args

def main_worker(idx: int, num_gpus: int, distributed: bool, args: argparse.Namespace):
    device = torch.device(args.device[idx])
    if device != 'cpu':
        torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:29500',
            world_size=num_gpus, rank=idx)
        
    batch_size = int(args.batch_size / num_gpus)

    #Get the data
    if args.dataset == 'mnist':
        train_loader, valid_loader = mnist_train_loader(batch_size=batch_size,
            device=device, distributed=distributed)
    elif args.dataset == 'cifar':
        train_loader, valid_loader = cifar_train_loader(batch_size=batch_size,
            device=device, distributed=distributed, augs=args.augs)
    else:
        train_loader, valid_loader = imagenet_train_loader(batch_size=batch_size,
            distributed=distributed)

    logger = Logger('teacher', args.dataset, args)
    one_channel = args.dataset == 'mnist'
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    if args.model == 'resnet18':
        model = ResNet18(one_channel=one_channel, num_classes=num_classes)
    elif args.model == 'resnet50':
        model = ResNet50(one_channel=one_channel, num_classes=num_classes) 
    else:
        model = ResNet152(one_channel=one_channel, num_classes=num_classes)

    model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    #Train the model
    train_sup(model, train_loader, valid_loader, device=device,
        loss_function=nn.CrossEntropyLoss(), epochs=args.epochs, lr=args.lr, 
        optimizer_choice=args.optimizer, scheduler_choice=args.scheduler, 
        patience=args.patience, early_stop=args.early_stop, 
        log_interval=args.log_interval, logger=logger, rank=idx, num_devices=num_gpus)


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

    num_gpus = len(args.device)
    #If we are doing distributed computation over multiple GPUs
    if num_gpus > 1:
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus, True, args))
    else:
        main_worker(0, 1, False, args)

if __name__ == '__main__':
    main()