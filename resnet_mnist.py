#Code based on tutorial by Marcin Zablocki and PyTorch example code
#https://zablo.net/blog/post/pytorch-resnet-mnist-jupyter-notebook-2021/
#https://github.com/pytorch/examples/blob/master/mnist/main.py

import argparse
import torch
import numpy as np
from torchvision.models import resnet18
from torch import nn
from mnist import mnist_train_loader
from training import train_sup
from logger import Logger

class ResNet18MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        #Set number of input channels to 1 (since MNIST images are greyscale)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x):
        return self.model(x)

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
                        help='Patience used in Plateau scheduler.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--early-stop', type=int, default=5, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    args = parser.parse_args()

    return args

def main():
    parser = argparse.ArgumentParser(description='ResNet-18 for MNIST')
    args = get_args(parser)

    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    logger = Logger('teacher', 'mnist', args)

    #Initialize the model
    model = ResNet18MNIST()

    #Get the data
    train_loader, valid_loader = mnist_train_loader(train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size, device=args.device)

    #Train the model
    train_sup(model, train_loader, valid_loader, device=args.device,
        train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size, 
        loss_function=nn.CrossEntropyLoss(), epochs=args.epochs, lr=args.lr,
        optimizer_choice=args.optimizer, patience=args.patience, early_stop=args.early_stop, 
        log_interval=args.log_interval, logger=logger, save_path=logger.get_model_path(), 
        plots_dir=logger.get_plots_dir())

if __name__ == '__main__':
    main()
