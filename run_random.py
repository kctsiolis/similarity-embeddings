"""Train a linear classifier on top of a randomly initialized feature embedder.

Embedders supported:
    ResNet18
    ResNet50

Datasets supported:
    MNIST
    CIFAR-10

"""

import argparse
import torch
import numpy as np
from torch import nn
from models import ResNet18, Embedder, Classifier
from mnist import mnist_train_loader
from cifar import cifar_train_loader
from imagenet import imagenet_train_loader
from training import train_sup
from logger import Logger

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
        help='Input batch size for training (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=1000, metavar='N',
        help='Input batch size for validation (default:1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='Input batch size for testing (default:1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                        help='Choice of optimizer for training.')
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine'], default='plateau',
                        help='Choice of scheduler for training.')
    parser.add_argument('--patience', type=int,
                        help='Patience used in Plateau scheduler.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--early-stop', type=int, default=5, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'],
                        help='Choice of model.')
    args = parser.parse_args()

    return args

def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Linear Classification on Top of Random Embedder')
    args = get_args(parser)

    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    #Get the data
    if args.dataset == 'mnist':
        train_loader, valid_loader = mnist_train_loader(train_batch_size=args.train_batch_size,
            valid_batch_size=args.valid_batch_size, device=args.device)
    elif args.dataset == 'cifar':
        train_loader, valid_loader = cifar_train_loader(train_batch_size=args.train_batch_size,
            valid_batch_size=args.valid_batch_size, device=args.device, augs=args.augs)
    else:
        train_loader, valid_loader = imagenet_train_loader(batch_size=args.train_batch_size)

    logger = Logger('random', args.dataset, args)
    one_channel = args.dataset == 'mnist'
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    #Randomly initialized ResNet18 with frozen embedder, learnable linear layer
    model = Classifier(Embedder(ResNet18(one_channel=one_channel, num_classes=num_classes)))

    #Train the model
    train_sup(model, train_loader, valid_loader, device=args.device,
        loss_function=nn.CrossEntropyLoss(), epochs=args.epochs, lr=args.lr, 
        optimizer_choice=args.optimizer, scheduler_choice=args.scheduler, patience=args.patience, 
        early_stop=args.early_stop, log_interval=args.log_interval, logger=logger)

if __name__ == '__main__':
    main()
