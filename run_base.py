import torch
import numpy as np
from training.loaders import get_loader

def get_base_args(parser):
    """Collect command line arguments common to all runners."""
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'tiny_imagenet'], metavar='D',
                        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--device', type=str, metavar='D', default='cpu',
                        help='Specify device to train on.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='sgd',
                        help='Choice of optimizer for training (default: sgd).')
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine', 'exponential'], default='cosine',
                        help='Choice of scheduler for training.')
    parser.add_argument('--plateau-patience', type=int, default=5,
                        help='Patience used in Plateau scheduler.')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--train-set-fraction', type=float, default=1.0,
                        help='Fraction of training set to train on (default: 1.0).')
    parser.add_argument('--lr-warmup-iters', type=int, default=0,
                        help='Number of iterations (batches) over which to perform learning rate warmup (default: 0).')
    parser.add_argument('--early-stop', type=int, default=50, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='Frequency of loss logging in terms of number of iterations (default: 10).')
    parser.add_argument('--save-each-epoch', action='store_true',
                        help='Save model at each epoch rather than overwriting model each time')
    parser.add_argument('--plot-interval', type=int,
                        help='Number of iterations between updates of loss plot.')                                         
    parser.add_argument('--no-save', action='store_true',
                        help='Don\'t save the log, plots, and model from this run.')    
    parser.add_argument('--train-subset-indices-path', type = str)
                        

    return parser

def run_base(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    device = torch.device(args.device)

    # Get the data
    train_loader, val_loader, num_classes = get_loader(args)

    return train_loader, val_loader, num_classes, device