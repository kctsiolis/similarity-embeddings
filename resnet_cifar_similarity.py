import argparse
from resnet_mnist import ResNet18MNIST
from resnet_cifar_distillation import ResNet18Embedder
from cifar import cifar_train_loader
from training import train_similarity
import torch
import numpy as np
from torchvision.models import resnet18
from torch import nn
from logger import Logger

class ResNet18NormalizedEmbedder(ResNet18Embedder):
    def __init__(self, model):
        super().__init__(model)
        self.final_normalization = nn.BatchNorm1d(512, affine=False)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.final_normalization(x) #Normalize the output
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
    parser.add_argument('--temp', type=float, default=1.0,
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
    parser = argparse.ArgumentParser(description='ResNet-18 for MNIST')
    args = get_args(parser)

    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    logger = Logger('similarity', 'cifar', args)

    if args.model == 'resnet18':
        model = ResNet18NormalizedEmbedder(resnet18(num_classes=10))
    else:
        raise ValueError('Invalid model choice.')

    train_loader, valid_loader = cifar_train_loader(train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size, device=args.device)

    if args.loss == 'mse':
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.KLDivLoss(reduction='batchmean')

    train_similarity(model, train_loader, valid_loader, device=args.device, augmentation=args.augmentation,
        alpha_max=args.alpha_max, train_batch_size=args.train_batch_size, 
        valid_batch_size=args.valid_batch_size, loss_function=loss_function, epochs=args.epochs, 
        lr=args.lr, optimizer_choice=args.optimizer, patience=args.patience, early_stop=args.early_stop, 
        log_interval=args.log_interval, logger=logger, cosine=args.cosine, temp=args.temp)

if __name__ == '__main__':
    main()
