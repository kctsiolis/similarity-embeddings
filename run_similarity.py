"""Train similarity-based embeddings from data augmentation.

The embeddings are trained so that the dot product between an image's 
embedding and the embedding of its augmented version reflect a pre-defined
notion of similarity. 

Models supported:
    ResNet18 (imported from PyTorch)

Datasets supported:
    MNIST
    CIFAR-10

Loss functions supported:
    MSE
    KL Divergence

"""

import argparse
import torch
import numpy as np
from torch import nn
from loaders import dataset_loader
from training import train_similarity
from models import get_model
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
    parser.add_argument('--cosine', action='store_true',
                        help='Use cosine similarity in the distillation loss.')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18'],
                        help='Choice of model.')
    parser.add_argument('--loss', type=str, choices=['mse', 'kl'], default='mse',
                        help='Type of loss function to use.')
    parser.add_argument('--temp', type=float, default=0.01,
                        help='Temperature in sigmoid function converting similarity score to probability.')
    parser.add_argument('--augmentation', type=str, choices=['blur-sigma', 'blur-kernel'], default='blur-sigma',
                        help='Augmentation to use.')
    parser.add_argument('--alpha-max', type=int, default=15,
                        help='Largest possible augmentation strength.')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Parameter of similarity probability function p(alpha).')

    args = parser.parse_args()

    return args

def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Similarity-based Embedding Learning')
    args = get_args(parser)

    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    #Get the data and initialize the model
    train_loader, valid_loader = dataset_loader(args.dataset,
        args.batch_size, device)

    logger = Logger('similarity', args.dataset, args)
    one_channel = args.dataset == 'mnist'
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    if args.loss == 'mse':
        loss_function = nn.MSELoss()
        model = get_model(args.model, one_channel=one_channel, num_classes=num_classes)
    else:
        loss_function = nn.KLDivLoss(reduction='batchmean')
        model = get_model(args.model, one_channel=one_channel, num_classes=num_classes, 
            batchnormalize=True)

    train_similarity(model, train_loader, valid_loader, device=device, 
        augmentation=args.augmentation, alpha_max=args.alpha_max, 
        loss_function=loss_function, epochs=args.epochs, lr=args.lr, 
        optimizer_choice=args.optimizer, scheduler_choice=args.scheduler, patience=args.patience, 
        early_stop=args.early_stop, log_interval=args.log_interval, logger=logger, 
        cosine=args.cosine, temp=args.temp)

if __name__ == '__main__':
    main()
