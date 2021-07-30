"""Load a pre-trained embedder and train a linear classifier on top.

The pre-trained embedder must be stored locally. Specify the path
with the "load-path" argument.

Embedders supported:
    Basic CNN
    ResNet18

Datasets supported:
    MNIST
    CIFAR-10
    
"""

import torch
import argparse
import torch
import numpy as np
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from models import get_model, Classifier
from loaders import dataset_loader
from training import train_sup
from logger import Logger

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    parser.add_argument('--early-stop', type=int, default=20, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load-path', type=str,
                        help='Path to the distilled "student" model.')
    parser.add_argument('--device', type=str, default=["cpu"], nargs='+',
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    parser.add_argument('--model', type=str, default='resnet18_embedder', choices=['cnn', 
                        'resnet18_embedder', 'resnet50_embedder', 'resnet50_simclr'],
                        help='Type of model for the embedder.')
    args = parser.parse_args()

    return args

def main_worker(idx: int, num_gpus: int, distributed: bool, args: argparse.Namespace):
    device = torch.device(args.device[idx])
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:29500',
            world_size=num_gpus, rank=idx)

    batch_size = int(args.batch_size / num_gpus)

    #Get the data
    train_loader, valid_loader = dataset_loader(args.dataset,
        batch_size, device, distributed)

    logger = Logger('linear_classifier', args.dataset, args, save=(idx == 0))
    one_channel = args.dataset == 'mnist'
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    if args.load_path is None:
        return ValueError('Path to the embedder is required.')

    embedder = get_model(args.model, load=True, load_path=args.load_path,
        one_channel=one_channel, num_classes=num_classes, get_embedder=True)     
    model = Classifier(embedder, num_classes=num_classes)
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
    parser = argparse.ArgumentParser(description='Linear Classification')
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