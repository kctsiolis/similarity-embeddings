"""Train a student to emulate the similarities between the teacher's embeddings.

Some code taken from Torch tutorial on classification for CIFAR-10
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Teachers supported:
    ResNet18

Students supported:
    ResNet18
    Basic CNN

Datasets supported:
    MNIST
    CIFAR-10

"""

import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import torch.nn as nn
from loaders import dataset_loader
from models import get_model
from training import train_distillation
from logger import Logger

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                        help='Choice of optimizer for training.')
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine'], default='plateau',
                        help='Choice of scheduler for training.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience used in the Plateau scheduler.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--early-stop', type=int, default=10, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load-path', type=str,
                        help='Path to the teacher model.')
    parser.add_argument('--device', type=str, default=["cpu"], nargs='+',
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    parser.add_argument('--teacher-model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50_pretrained', 'simclr_pretrained'],
                        help='Choice of student model.')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'cnn'],
                        help='Choice of student model.')
    parser.add_argument('--cosine', action='store_true',
                        help='Use cosine similarity in the distillation loss.')

    args = parser.parse_args()

    return args

def main_worker(idx: int, num_gpus: int, distributed: bool, args: argparse.Namespace):
    device = torch.device(args.device[idx])
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:29501',
            world_size=num_gpus, rank=idx)
        
    batch_size = int(args.batch_size / num_gpus)

    #Get the data
    train_loader, valid_loader = dataset_loader(args.dataset,
        batch_size, device, distributed)

    logger = Logger('distillation', args.dataset, args, save=(idx == 0))
    one_channel = args.dataset == 'mnist'
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    teacher = get_model(args.teacher_model, load=True, load_path=args.load_path, 
        one_channel=one_channel, num_classes=num_classes, get_embedder=True)
    student = get_model(args.model, one_channel=one_channel, num_classes=num_classes)
    student.to(device)
    teacher.to(device)

    if distributed:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[device])
        teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[device])

    train_distillation(student, teacher, train_loader, valid_loader, device=device, 
        loss_function=nn.MSELoss(), epochs=args.epochs, lr=args.lr, optimizer_choice=args.optimizer,
        scheduler_choice=args.scheduler, patience=args.patience, early_stop=args.early_stop, 
        log_interval=args.log_interval, logger=logger, cosine=args.cosine, rank=idx,
        num_devices=num_gpus)

def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Similarity-based Knowledge Distillation')
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