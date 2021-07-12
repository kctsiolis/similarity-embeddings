"""Plot model similarities between data and augmented data.

Given a pre-trained embedder for image data, compute the 
dot product (or cosine) similarity between the model's representation
of images in the data and their augmented version (with a fixed 
augmentation intensity). Plot a histogram of the similarities.

"""

import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from data_augmentation import augment
from training import get_model_similarity
from models import Embedder, ResNet18
from mnist import mnist_train_loader
from cifar import cifar_train_loader

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
    plt.title('Model similarities for {} with intensity {}'.format(augmentation, alpha))
    plt.savefig(save_path)

def get_args(parser):
    parser.add_argument('--load-path', type=str,
                        help='Path to the model.')   
    parser.add_argument('--model', type=str, choices=['resnet18'], default='resnet18',
                        help='Type of model being evaluated.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], default='cifar',
                        help='Dataset model was trained on.')
    parser.add_argument('--augmentation', type=str, choices=['blur-sigma'], default='blur-sigma',
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

    #Get the data
    if args.dataset == 'mnist':
        one_channel = True
        train_loader, valid_loader = mnist_train_loader(64, 1000, device=args.device)
    else:
        one_channel = False
        train_loader, valid_loader = cifar_train_loader(64, 1000, device=args.device)

    #For now, only ResNet18 is supported
    device = torch.device(args.device)
    model = ResNet18(one_channel=one_channel)
    model.load_state_dict(torch.load(args.load_path), strict=False)
    model = Embedder(model)
    model = model.to(device)

    model_sims = collect_sims(model, valid_loader, args.augmentation,
        device, args.alpha, args.cosine)

    print('Minimum: {:.4f}'.format(min(model_sims)))
    print('Maximum: {:.4f}'.format(max(model_sims)))
    print('1st Percentile: {:.4f}'.format(np.percentile(model_sims, 1)))
    print('2.5th Percentile: {:.4f}'.format(np.percentile(model_sims, 2.5)))
    print('5th Percentile: {:.4f}'.format(np.percentile(model_sims, 5)))

    if args.save_path is not None:
        plot_sims(model_sims, args.augmentation, args.alpha, args.save_path)

if __name__ == '__main__':
    main()