"""Evaluate a model on a classification task.

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
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from torch.utils import data
from models import get_model
from training import predict
from loaders import dataset_loader


def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar','cifar100', 'imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'] ,metavar='D',
        help='Choice of either training, validation, or test subset (where applicable).')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='Batch size (default: 64)')
    parser.add_argument('--train-set-fraction', type=float, default=1.0,
                        help='Fraction of training set to train on.')
    parser.add_argument('--validate', action='store_true',
                        help='Evaluate on a held out validation set (as opposed to the test set).')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Name of CUDA device being used (if any).')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Choice of model.')
    parser.add_argument('--load-path', type=str,
                        help='Path to the teacher model.')
    parser.add_argument('--precision', type = str,choices = ['autocast','32','16','8'],default = '32',
                        help='Evaluate models in half precision')        
    parser.add_argument('--calculate-confusion', action='store_true',
                    help='Calculate confusion matrix along with accuracy.')                                                                    

    args = parser.parse_args()

    return args    

def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Training the teacher model')
    args = get_args(parser)

    if args.precision in ['8','4']:
        args.device = 'cpu'
    
    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    loader1, loader2 = dataset_loader(
        args.dataset, args.batch_size, 
        train_set_fraction=args.train_set_fraction, 
        validate=args.validate)
            
    one_channel = args.dataset == 'mnist'
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    model = get_model(
        args.model, load=True, load_path=args.load_path, 
        one_channel=one_channel, num_classes=num_classes)            
    model.to(device)

    if args.split == 'train':
        metrics = predict(model, device, loader1, nn.CrossEntropyLoss(), args.precision, args.calculate_confusion)
    elif args.split == 'val':
        metrics = predict(model, device, loader2, nn.CrossEntropyLoss(), args.precision, args.calculate_confusion)
    else:
        metrics = predict(model, device, loader2, nn.CrossEntropyLoss(), args.precision, args.calculate_confusion)

    print('Loss: {:.6f}'.format(metrics[0]))
    print('Top-1 Accuracy: {:.2f}'.format(metrics[1]))
    print('Top-5 Accuracy: {:.2f}'.format(metrics[2]))

    if args.calculate_confusion:
        sn.heatmap(metrics[3],annot=True)
        plt.savefig('confusion_matrix.png')
        plt.close()
    
if __name__ == '__main__':
    main()