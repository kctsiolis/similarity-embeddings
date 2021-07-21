"""Plot model similarities between input images.

Given a pre-trained embedder for image data, compute the 
dot product (or cosine) similarity between the model's representation
of images in the data and their augmented version (with a fixed 
augmentation intensity). Plot a histogram of the similarities.

Alternatively, if the augmentation is chosen to be "none", then 
dot products (or cosine similarities) are computed between
every pair of images in the original data.

"""

import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from data_augmentation import augment
from embeddings import compute_similarity_matrix, normalize_embeddings
from training import get_model_similarity, get_embeddings, predict
from models import Embedder, ResNet18, ResNet50, ResNetSimCLR
from mnist import mnist_train_loader
from cifar import cifar_train_loader

def get_args(parser):
    parser.add_argument('--load-path', type=str,
                        help='Path to the model.')   
    parser.add_argument('--model', type=str, choices=['resnet18_classifier', 'resnet18_pretrained',
                        'resnet18_embedder', 'resnet50_pretrained', 'simclr', 'simclr_pretrained'], 
                        default='resnet18',
                        help='Type of model being evaluated.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], default='cifar',
                        help='Dataset model was trained on.')
    parser.add_argument('--augmentation', type=str, choices=['blur-sigma', 'none'], default='blur-sigma',
                        help='Type of augmentation to use.') 
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Augmentation intensity.')
    parser.add_argument('--cosine', action='store_true',
                        help='Set this option to use cosine similarity.')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to use.')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save plot.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    
    args = parser.parse_args()
    return args

def collect_sims(model: nn.Module, loader: torch.utils.data.DataLoader, 
    augmentation: str, device: torch.device, alpha: int,
    cosine: bool = True) -> list:
    """Compute the model similarities for the entire dataset.

    For each image in the data, get an augmented version (using the
    fixed intensity alpha). Then, get the model's similarity score
    between the original and augmented image.

    Args:
        model: The model being used.
        loader: The data.
        augmentation: The type of data augmentation.
        device: The device the model and data are loaded to.
        alpha: Transformation intensity.
        cosine: Whether or not to use cosine similarity.

    Returns:
        The computed model similarities.

    """
    model_sims = []
    model.eval()
    with torch.no_grad():
        for data, _ in tqdm(loader):
            data = data.to(device)
            augmented_data, _ = augment(data, augmentation, device, alpha, 0.2, False)
            model_sims += get_model_similarity(model, data, augmented_data, cosine).tolist()

    return model_sims

def plot_sims(sims: list, augmentation: str, alpha: float, 
    save_path: str) -> None:
    """Plot a histogram of the model similarities.

    Args:
        sims: The similarities.
        augmentation: The type of data augmentation.
        alpha: The augmentation intensity.
        save_path: Path to save the histogram.

    """
    plt.figure()
    plt.hist(sims)
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    if augmentation != 'none':
        plt.title('Model similarities for {} with intensity {}'.format(augmentation, alpha))
    plt.savefig(save_path)

def summary(sims):
    print('Minimum: {:.4f}'.format(min(sims)))
    print('Maximum: {:.4f}'.format(max(sims)))
    print('Mean: {:.4f}'.format(np.mean(sims)))
    print('Standard deviation: {:.4f}'.format(np.std(sims)))
    print('1st Percentile: {:.4f}'.format(np.percentile(sims, 1)))
    print('2.5th Percentile: {:.4f}'.format(np.percentile(sims, 2.5)))
    print('5th Percentile: {:.4f}'.format(np.percentile(sims, 5)))
    print('25th Percentile: {:.4f}'.format(np.percentile(sims, 25)))
    print('Median: {:.4f}'.format(np.percentile(sims, 50)))
    print('75th Percentile: {:.4f}'.format(np.percentile(sims, 75)))

def main():
    """Load the data, compute similarities, and plot them."""
    parser = argparse.ArgumentParser(description='Plotting model similarities.')
    args = get_args(parser)

    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    #Get the data
    if args.dataset == 'mnist':
        one_channel = True
        _, valid_loader = mnist_train_loader(64, device=device)
    else:
        one_channel = False
        _, valid_loader = cifar_train_loader(64, device=device)

    if args.model == 'resnet18_classifier':
        model = ResNet18(one_channel=one_channel)
        try:
            model.load_state_dict(torch.load(args.load_path))
        except RuntimeError:
            model.model.load_state_dict(torch.load(args.load_path))
        model = Embedder(model)
    elif args.model == 'resnet18_pretrained':
        model = ResNet18(num_classes=1000, pretrained=True)
        model = Embedder(model)
    elif args.model == 'resnet18_embedder':
        model = Embedder(ResNet18())
        model.load_state_dict(torch.load(args.load_path))
    elif args.model == 'resnet50_pretrained':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
        model = Embedder(model, dim=64)
    elif args.model == 'simclr':
        checkpoint = torch.load(args.load_path)
        model = ResNetSimCLR(base_model='resnet18', out_dim=128)
        model.load_state_dict(checkpoint['state_dict'])
        model = Embedder(model)
    else:
        checkpoint = torch.load(args.load_path)
        model = ResNet50(num_classes=1000)
        model.model.load_state_dict(checkpoint['state_dict'])
        model = Embedder(model)

    model = model.to(device)

    if args.augmentation == 'none':
        model_embs, labels = get_embeddings(model, device, valid_loader, model.get_dim())
        if args.cosine:
            model_embs = normalize_embeddings(model_embs)

        sorted_embs = []
        c = int(np.max(labels) + 1)

        for i in range(c):
            sorted_embs.append(model_embs[np.where(labels == i)])

        intra_class_sims = []
        inter_class_sims = []
        for i in range(c):
            intra_class_sims += np.matmul(sorted_embs[i], np.transpose(sorted_embs[i])).flatten().tolist()
            if i < c - 1: 
                for j in range(i+1, c):
                    inter_class_sims += np.matmul(sorted_embs[i], np.transpose(sorted_embs[j])).flatten().tolist()

        print('INTRA CLASS SIMS:')
        summary(intra_class_sims)

        print('INTER CLASS SIMS:')
        summary(inter_class_sims)

        print('OVERALL SIMS:')
        model_sims = intra_class_sims + 2 * inter_class_sims
        summary(model_sims)
    else:
        model_sims = collect_sims(model, valid_loader, args.augmentation,
            device, args.alpha, args.cosine)

    summary(model_sims)

    if args.save_path is not None:
        plot_sims(model_sims, args.augmentation, args.alpha, args.save_path)

if __name__ == '__main__':
    main()