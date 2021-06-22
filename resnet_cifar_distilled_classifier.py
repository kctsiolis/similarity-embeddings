import torch
import argparse
import torch
import numpy as np
from torch import nn
from torchvision.models import resnet18
from resnet_cifar_distillation import ResNet18MNISTEmbedder
from resnet_cifar_distillation import ConvNetEmbedder
from resnet_mnist_distilled_classifier import ResNet18DistilledClassifier
from cifar import cifar_train_loader
from training import train_sup
from logger import Logger

#Class for a classifier on top of a ConvNet
class ConvNetDistilledClassifier(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.embedder = student
        #Freeze the embedding layer
        for param in self.embedder.parameters():
            param.requires_grad = False
        #Classification layer
        self.linear_layer = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = self.embedder(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)

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
    parser.add_argument('--load-path', type=str,
                        help='Path to the distilled "student" model.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    parser.add_argument('--small-student', action='store_true',
                        help='Use a small student model (vanilla CNN with 2 conv and pooling layers).')
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

    logger = Logger('linear_classifier', 'cifar', args)

    if args.load_path is None:
        return ValueError('Path to teacher network is required.')

    if args.small_student:
        student = ConvNetEmbedder()
        student.load_state_dict(torch.load(args.load_path))
        model = ConvNetDistilledClassifier(student)
    else:
        student = ResNet18Embedder(resnet18)
        student.load_state_dict(torch.load(args.load_path))        
        model = ResNet18DistilledClassifier(student)
    
    #Get the data
    train_loader, valid_loader = cifar_train_loader(train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size, device=args.device)

    #Train the model
    train_sup(model, train_loader, valid_loader, device=args.device,
        train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size, 
        loss_function=nn.CrossEntropyLoss, epochs=args.epochs, lr=args.lr, optimizer_choice=args.optimizer,
        patience=args.patience, early_stop=args.early_stop, log_interval=args.log_interval, 
        logger=logger)

if __name__ == '__main__':
    main()