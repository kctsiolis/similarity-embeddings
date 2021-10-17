"""Run any of the training modes on an image dataset.

Modes supported:
    Teacher
    Distillation
    Similarity
    Linear Classifier
    Random

Models supported:
    ResNet18
    ResNet50
    ResNet152

Datasets supported:
    MNIST
    CIFAR-10
    ImageNet

"""

import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from models import get_model, Classifier,WrapWithProjection
from loaders import dataset_loader
from training import get_trainer
from logger import Logger

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--mode', type=str, choices=['teacher', 'distillation', 'similarity',
        'linear_classifier', 'random'] ,metavar='D',
        help='Training mode.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'cifar100', 'imagenet', 'tiny_imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='sgd',
                        help='Choice of optimizer for training (default: sgd).')
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine'], default='cosine',
                        help='Choice of scheduler for training.')
    parser.add_argument('--plateau-patience', type=int, default=5,
                        help='Patience used in Plateau scheduler.')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--train-set-fraction', type=float, default=1.0,
                        help='Fraction of training set to train on (default: 1.0).')
    parser.add_argument('--validate', action='store_true',
                        help='Evaluate on a held out validation set (as opposed to the test set).')
    parser.add_argument('--lr-warmup-iters', type=int, default=0,
                        help='Number of iterations (batches) over which to perform learning rate warmup (default: 0).')
    parser.add_argument('--early-stop', type=int, default=10, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='Frequency of loss logging in terms of number of iterations (default: 10).')
    parser.add_argument('--save-each-epoch', action = 'store_true',
                        help='Save model at each epoch rather than override model each time')                        
    parser.add_argument('--plot-interval', type=int,
                        help='Number of iterations between updates of loss plot.')
    parser.add_argument('--device', type=str, nargs='+', default=['cpu'],
                        help='Name of CUDA device(s) being used (if any). Otherwise will use CPU. \
                            Can also specify multiple devices (separated by spaces) for multiprocessing.')
    parser.add_argument('--load-path', type=str,
                        help='Path to the teacher model.')
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='Choice of teacher model (for distillation only).')
    parser.add_argument('--student-model', type=str, 
                        help='Choice of student model.')
    parser.add_argument('--cosine', action='store_true',
                        help='Use cosine similarity in the distillation loss.')
    parser.add_argument('--distillation-type', type=str, choices=['similarity-based', 'class-probs'],
                        default='similarity-based',
                        help='Use cosine similarity in the distillation loss.')
    parser.add_argument('-c', type=float, default=0.5,
                        help='Weighing of soft target and hard target loss in class-probs distillation.')
    parser.add_argument('--augmentation', type=str, choices=['blur', 'jitter', 'crop'], default=None,
                        help='Data augmentation to use for similarity embedding training.')
    parser.add_argument('--alpha-max', type=float, default=1.0,
                        help='Largest possible augmentation strength.')
    parser.add_argument('--kernel-size', type=int, default=None,
                        help='Kernel size parameter for Gaussian blur.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Parameter of similarity probability function p(alpha).')
    parser.add_argument('--temp', type=float, default=0.01,
                        help='Temperature in sigmoid function converting similarity score to probability.')
    parser.add_argument('--wrap-in-projection', action = 'store_true',
                        help='Wrap the teacher model in a random projection (For distillation only)')                        
    parser.add_argument('--projection-dim', type = int,default = None,
                        help='Dimension to of projection to wrap the teacher model in')                        

    args = parser.parse_args()

    return args

def main_worker(idx: int, num_gpus: int, distributed: bool, args: argparse.Namespace):
    device = torch.device(args.device[idx])

    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:29501',
            world_size=num_gpus, rank=idx)
        
    batch_size = int(args.batch_size / num_gpus)

    #Get the data
    train_loader, val_loader = dataset_loader(
        args.dataset, batch_size, args.train_set_fraction, 
        args.validate, distributed)

    logger = Logger(args, save=(idx == 0))
    one_channel = args.dataset == 'mnist'
    if args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        # Cifar10  or MNIST
        num_classes = 10
    
    if args.mode == 'teacher' or args.mode == 'random':
        model = get_model(args.student_model, one_channel=one_channel, num_classes=num_classes)
    elif args.mode == 'linear_classifier':
        model = get_model(
            args.student_model, load=True, load_path=args.load_path,
            one_channel=one_channel, num_classes=num_classes)
    elif args.mode == 'distillation' and args.distillation_type == 'class-probs':
        model = get_model(args.student_model, one_channel=one_channel, num_classes=num_classes)
    else:
        model = get_model(args.student_model, one_channel=one_channel, get_embedder=True)
    
    if args.mode == 'linear_classifier' or args.mode == 'random':
        model = Classifier(model, num_classes=num_classes)

    model.to(device)

    if args.mode == 'distillation':
        get_embedder = args.distillation_type == 'similarity-based'
        get_embedder = False        
        teacher = get_model(
            args.teacher_model, load=True, load_path=args.load_path, 
            one_channel=one_channel, num_classes=num_classes, 
            get_embedder=get_embedder).to(device) 
        if args.wrap_in_projection:
            teacher = WrapWithProjection(teacher,teacher.dim,args.projection_dim).to(device)        
    else:
        teacher = None

    if distributed:
        model = DistributedDataParallel(model, device_ids=[device])
        if args.mode == 'distillation':
            teacher = DistributedDataParallel(teacher, device_ids=[device])

    trainer = get_trainer(args.mode, model, teacher, train_loader, val_loader, device, logger, idx, args)

    trainer.train()

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