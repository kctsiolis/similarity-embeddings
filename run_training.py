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
    ResNet152arg

Datasets supported:
    CIFAR-10
    CIFAR-100
    TinyImageNet
    ImageNet

"""

import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from models import get_model, Classifier, WrapWithProjection, EmbedderAndLogits
from loaders import dataset_loader
from training import get_trainer
from logger import Logger


def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--mode', type=str, choices=['teacher', 'distillation', 'linear_classifier'], metavar='D',
                        help='Training mode.')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'tiny_imagenet'], metavar='D',
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
    parser.add_argument('--lr-warmup-iters', type=int, default=0,
                        help='Number of iterations (batches) over which to perform learning rate warmup (default: 0).')
    parser.add_argument('--early-stop', type=int, default=10, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='Frequency of loss logging in terms of number of iterations (default: 10).')
    parser.add_argument('--save-each-epoch', action='store_true',
                        help='Save model at each epoch rather than overwriting model each time')
    parser.add_argument('--plot-interval', type=int,
                        help='Number of iterations between updates of loss plot.')
    parser.add_argument('--teacher-path', type=str,
                        help='Path to the teacher model.')
    parser.add_argument('--student-path', type=str,
                        help='Path to the student model.')
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='Choice of teacher model.')
    parser.add_argument('--student-model', type=str,
                        help='Choice of student model.')
    parser.add_argument('--project-embedder', action='store_true',
                        help='Add a projection head to the embedder.')
    parser.add_argument('--distillation-loss', type=str, choices=['similarity-based', 'similarity-weighted', 'kd'],
                        default='similarity-based',
                        help='Loss used for distillation.')
    parser.add_argument('--augmented-distillation', action='store_true',
                        help='Whether or not to use data augmentation in distillation.')
    parser.add_argument('-c', type=float, default=0.5,
                        help='Weighing of soft target and hard target loss in Hinton\'s KD.')
    parser.add_argument('--wrap-in-projection', action='store_true',
                        help='Wrap the teacher model in a random projection (For distillation only)')
    parser.add_argument('--projection-dim', type=int, default=None,
                        help='Dimension to of projection to wrap the teacher model in')
    parser.add_argument('--margin', action='store_true',
                        help='(For cosine distillation only) Should angular margin be applied ')
    parser.add_argument('--margin-value', type = float,default = 0.5,
                        help='If [margin] is selected what should it be set to (Default 0.5)')     
    parser.add_argument('--truncate-model', action = 'store_true',
                        help='Should we truncate the (student) model when training a linear classifier?')                        
    parser.add_argument('--truncation-level', type =int,
                        help='How many layers to remove to form the truncated (student) model')                                                     
    parser.add_argument('--no-save', action='store_true',
                        help='Don\'t save the log, plots, and model from this run.')                                                

    args = parser.parse_args()

    return args

def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Training the student model.')
    args = get_args(parser)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Get the data
    train_loader, val_loader, num_classes = dataset_loader(args)

    logger = Logger(args)

    load_student = (args.student_path is not None)

    if args.mode == 'teacher':
        model = get_model(args.student_model,
                          num_classes=num_classes)
    elif args.mode == 'linear_classifier':
        model = get_model(
            args.student_model, load=True, load_path=args.student_path,
            num_classes=num_classes, truncate_model=args.truncate_model, truncation_level=args.truncation_level)
    elif args.mode == 'distillation':
        get_embedder = args.distillation_loss != 'kd'
        model = get_model(args.student_model, load=load_student,
                          load_path=args.student_path,
                          get_embedder=get_embedder, num_classes=num_classes)
        teacher = get_model(
            args.teacher_model, load=True, load_path=args.teacher_path,
            num_classes=num_classes, get_embedder=get_embedder).to(device)
        if args.wrap_in_projection:
            teacher = WrapWithProjection(
                teacher, teacher.dim, args.projection_dim).to(device)
    elif args.mode == 'weighted-distillation':
        get_embedder = args.distillation_type != 'kd'
        model = get_model(args.student_model, load=load_student,
                          load_path=args.student_path,
                          get_embedder=get_embedder, num_classes=num_classes)
        teacher = get_model(
            args.teacher_model, load=True, load_path=args.teacher_path,
            num_classes=num_classes, get_embedder=False)
        teacher = EmbedderAndLogits(teacher).to(device)        
    else:
        model = get_model(args.student_model, load=load_student,
                          load_path=args.student_path, get_embedder=True,project_embedder=args.project_embedder)

    if args.mode == 'linear_classifier':
        model = Classifier(model, num_classes=num_classes)

    model.to(device)

    if (args.mode == 'distillation') or (args.mode == 'weighted-distillation'):
        get_embedder = args.distillation_loss != 'kd'
    else:
        teacher = None

    trainer = get_trainer(args.mode, model, teacher,
                          train_loader, val_loader, device, logger, args)

    trainer.train()


if __name__ == '__main__':
    main()
