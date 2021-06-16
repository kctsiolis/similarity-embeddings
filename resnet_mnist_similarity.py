import argparse
from resnet_mnist import ResNet18MNIST
from resnet_mnist_distillation import ResNet18MNISTEmbedder
from mnist import mnist_train_loader
from training import train_similarity
import torch
from torch import nn

def get_args(parser):
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
        help='Input batch size for training (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=1000, metavar='N',
        help='Input batch size for validation (default:1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='Input batch size for testing (default:1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--early-stop', type=int, default=5, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load-path', type=str,
                        help='Path to the teacher model.')
    parser.add_argument('--save-path', type=str,
                        help='Path for saving the student model')
    parser.add_argument('--plots-dir', type=str,
                        help='Directory for saving loss and accuracy plots.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    parser.add_argument('--cosine', action='store_true',
                        help='Use cosine similarity in the distillation loss.')
    args = parser.parse_args()

    return args

def main():
    parser = argparse.ArgumentParser(description='ResNet-18 for MNIST')
    args = get_args(parser)

    model = ResNet18MNISTEmbedder(ResNet18MNIST())

    train_loader, valid_loader = mnist_train_loader(train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size, device=args.device)

    train_similarity(model, train_loader, valid_loader, device=args.device, augmentation='blur',
        seed=args.seed, train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size, 
        loss_function=nn.MSELoss, epochs=args.epochs, lr=args.lr, gamma=args.gamma, 
        early_stop=args.early_stop, log_interval=args.log_interval, save_path=args.save_path, 
        plots_dir=args.plots_dir, cosine=args.cosine)

if __name__ == '__main__':
    main()
