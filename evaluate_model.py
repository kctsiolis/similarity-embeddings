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
from torch.utils import data
from models import get_model
from loaders import dataset_loader
from training import predict

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'] ,metavar='D',
        help='Choice of either training, validation, or test subset (where applicable).')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='Batch size (default: 64)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Name of CUDA device being used (if any).')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Choice of model.')
    parser.add_argument('--load-path', type=str,
                        help='Path to the teacher model.')

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
    validate = args.split != 'test'

    loader1, loader2 = dataset_loader(args.dataset,
        args.batch_size, device, validate=validate)
            
    one_channel = args.dataset == 'mnist'
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    model = get_model(args.model, load=True, load_path=args.load_path, 
        one_channel=one_channel, num_classes=num_classes)
    model.to(device)

    if args.split == 'train':
        loss, acc = predict(model, device, loader1, nn.CrossEntropyLoss())
    elif args.split == 'val':
        loss, acc = predict(model, device, loader2, nn.CrossEntropyLoss())
    else:
        loss, acc = predict(model, device, loader2, nn.CrossEntropyLoss())

    print('Loss: {:.6f}'.format(loss))
    print('Accuracy: {:.2f}'.format(acc))

if __name__ == '__main__':
    main()