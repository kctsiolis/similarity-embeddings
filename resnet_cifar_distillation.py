#Some code taken from Torch tutorial on classification for CIFAR-10
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from cifar import cifar_train_loader
from training import train_distillation
from logger import Logger

class ResNet18Embedder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class ConvNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    #Get the embedding
    def embed(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x

def get_args(parser):
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
    parser.add_argument('--patience', type=int,
                        help='Patience used in the Plateau scheduler.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--early-stop', type=int, default=5, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load-path', type=str,
                        help='Path to the teacher model.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    parser.add_argument('--small-student', action='store_true',
                        help='Use a small student model (vanilla CNN with 2 conv and pooling layers).')
    parser.add_argument('--cosine', action='store_true',
                        help='Use cosine similarity in the distillation loss.')

    args = parser.parse_args()

    return args

def main():
    parser = argparse.ArgumentParser(description='Distilling a ResNet-18 for CIFAR-10')
    args = get_args(parser)

    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    logger = Logger('distillation', 'cifar', args)

    if args.small_student:
        student = ConvNetEmbedder()
    else:
        student = ResNet18Embedder(resnet18(num_classes=10))

    teacher = resnet18(num_classes=10)
    teacher.load_state_dict(torch.load(args.load_path))
    teacher = ResNet18Embedder(teacher)

    train_loader, valid_loader = cifar_train_loader(train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size, device=args.device)

    train_distillation(student, teacher, train_loader, valid_loader, device=args.device, 
    train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size, 
    loss_function=nn.MSELoss(), epochs=args.epochs, lr=args.lr, optimizer_choice=args.optimizer,
    patience=args.patience, early_stop=args.early_stop, log_interval=args.log_interval, logger=logger, 
    cosine=args.cosine)

if __name__ == '__main__':
    main()