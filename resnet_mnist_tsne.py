import argparse
import numpy as np 
import torch
import matplotlib.pyplot as plt
from mnist import mnist_train_loader
from resnet_mnist import ResNet18MNIST
from resnet_mnist_distillation import ResNet18MNISTEmbedder
from training import get_embeddings, get_labels
from embedding_analysis import tsne, tsne_plot

def get_args(parser):
    parser.add_argument('--load-path', type=str,
                        help='Path to the teacher model.')
    parser.add_argument('--save-path', type=str,
                        help='Path for saving the T-SNE embeddings.')
    parser.add_argument('--plots-path', type=str,
                        help='Path for saving the T-SNE plot.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    args = parser.parse_args()

    return args

def main():
    parser = argparse.ArgumentParser(description='T-SNE for ResNet-18 embeddings on MNIST')
    args = get_args(parser)

    train_loader, valid_loader = mnist_train_loader(train_batch_size=1000, 
        valid_batch_size=1000, device=args.device)

    model = ResNet18MNISTEmbedder(ResNet18MNIST())
    model.load_state_dict(torch.load(args.load_path))

    embs, labels = get_embeddings(model, args.device, train_loader, 512)
    tsne(embs, labels, args.save_path, 10, args.plots_path)

if __name__ == '__main__':
    main()