"""Plot model similarities between input images.

Given a pre-trained embedder for image data, compute the 
dot product (or cosine) similarity between the model's representation
of images in the data and their augmented version (with a fixed 
augmentation intensity). Plot a histogram of the similarities.

Alternatively, if the augmentation is chosen to be "none", then 
dot products (or cosine similarities) are computed between
every pair of images in the original data.

"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from training.data_augmentation import Augmentation, make_augmentation
from training.training import get_model_similarity
from models.models import get_model
from training.loaders import dataset_loader

def get_args(parser):
    parser.add_argument('--load-path', type=str,
                        help='Path to the model.')   
    parser.add_argument('--model', type=str, default='simclr_pretrained',
                        help='Model used to compute similarities.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='Batch size (default: 64)')
    parser.add_argument('--num-samples', type=int, default=None, metavar='N',
                        help='Number of data samples to use. Default is to use entire validation set.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'], default='imagenet',
                        help='Dataset to comptue similarities on.')
    parser.add_argument('--aug1', type=str, choices=['blur', 'jitter', 'crop'], default='blur',
                        help='First augmentation to apply.') 
    parser.add_argument('--alpha1', type=float, default=1.0,
                        help='Intensity of first augmentation.')
    parser.add_argument('--aug2', type=str, choices=['blur', 'jitter', 'crop'], default='blur',
                        help='Second augmentation to apply.')
    parser.add_argument('--alpha2', type=float, default=1.0,
                        help='Intensity of second augmentation.')
    parser.add_argument('--cosine', action='store_true',
                        help='Set this option to use cosine similarity.')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to use.')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    
    args = parser.parse_args()
    return args

def collect_composed_sims(
    model: nn.Module, loader: torch.utils.data.DataLoader, 
    aug1: Augmentation, aug2: Augmentation, device: torch.device,
    cosine: bool = True) -> list:
    """Compute the model similarities for the entire dataset.

    For each image in the data, get an augmented version (using the
    fixed intensity alpha). Then, get the model's similarity score
    between the original and augmented image.

    Args:
        model: The model being used.
        loader: The data.
        aug1: The first augmentation.
        aug2: The second augmentation.
        device: The device the model and data are loaded to.
        cosine: Whether or not to use cosine similarity.

    Returns:
        The computed model similarities.

    """
    model_sims = np.zeros((0))
    model.eval()
    with torch.no_grad():
        for data, _ in tqdm(loader):
            data = data.to(device)
            augmented_data, _ = aug1.augment(data)
            augmented_data, _ = aug2.augment(augmented_data)
            batch_sims = get_model_similarity(model, data, augmented_data, cosine).detach().cpu().numpy()
            model_sims = np.concatenate((model_sims, batch_sims))

    return model_sims

def plot_sims(sims: list, save_path: str) -> None:
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
    plt.savefig(save_path)

def summary(sims):
    mean = np.mean(sims)
    print('Mean: {:.4f}'.format(mean))
    std = np.std(sims)
    print('Standard deviation: {:.4f}'.format(std))

    return mean, std

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
    _, val_loader = dataset_loader(
        args.dataset, args.batch_size, 1.0, True, False)

    one_channel = True if args.dataset == 'mnist' else False
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    model = get_model(args.model, load=True, load_path=args.load_path,
        one_channel=one_channel, num_classes=num_classes,
        get_embedder=True)
    model = model.to(device)

    augmentation1 = make_augmentation(
        args.aug1, alpha_max=args.alpha1, device=device, random=False)
    augmentation2 = make_augmentation(
        args.aug2, alpha_max=args.alpha2, device=device, random=False)

    sims1 = collect_composed_sims(
        model, val_loader, augmentation1, augmentation2, device, args.cosine)
    mean1, std1 = summary(sims1)

    sims2 = collect_composed_sims(
        model, val_loader, augmentation2, augmentation1, device, args.cosine)
    mean2, std2 = summary(sims2)

    diff = sims1 - sims2
    mean_diff, std_diff = summary(diff)
    
    if args.save_dir is not None:
        save_path1 = os.path.join(args.save_dir, '{}_alpha1={}_{}_alpha2={}.png'.format(
            args.aug1, args.alpha1, args.aug2, args.alpha2))
        save_path2 = os.path.join(args.save_dir, '{}_alpha1={}_{}_alpha2={}.png'.format(
            args.aug2, args.alpha2, args.aug1, args.alpha1))
        save_path_diff = os.path.join(args.save_dir, 'difference_{}_alpha1={}_{}_alpha2={}.png'.format(
            args.aug1, args.alpha1, args.aug2, args.alpha2))
        plot_sims(sims1, save_path1)
        plot_sims(sims2, save_path2)
        plot_sims(diff, save_path_diff)

if __name__ == '__main__':
    main()