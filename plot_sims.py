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
from data_augmentation import make_augmentation
from embeddings import normalize_embeddings
from training import get_model_similarity, get_embeddings
from models import get_model
from loaders import dataset_loader

def get_args(parser):
    parser.add_argument('--load-path', type=str,
                        help='Path to the model.')   
    parser.add_argument('--model', type=str, choices=['resnet18_classifier', 'resnet18_pretrained',
                        'resnet18_embedder', 'resnet50_pretrained_cifar', 'simclr', 'simclr_pretrained'], 
                        default='resnet18',
                        help='Type of model being evaluated.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='Batch size (default: 64)')
    parser.add_argument('--num-samples', type=int, default=None, metavar='N',
                        help='Number of data samples to use. Default is to use entire validation set.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'], default='cifar',
                        help='Dataset model was trained on.')
    parser.add_argument('--augmentation', type=str, choices=['blur-sigma', 'color-jitter', 
                        'random-crop', 'none'], default='blur-sigma',
                        help='Type of augmentation to use.') 
    parser.add_argument('--alpha', type=float, default=[1.0], nargs='+',
                        help='Augmentation intensity.')
    parser.add_argument('--batchnormalize', action='store_true',
                        help='Set this option to use batch normalization to normalize the features.')
    parser.add_argument('--cosine', action='store_true',
                        help='Set this option to use cosine similarity.')
    parser.add_argument('--classifier', action='store_true',
                        help='Get similarities from classifier layer.')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to use.')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    
    args = parser.parse_args()
    return args

def collect_sims(model: nn.Module, loader: torch.utils.data.DataLoader, 
    augmentation: str, device: torch.device, alpha: float,
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
            augmented_data, _ = augmentation.augment(data)
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

def plot_mean(alphas: list, means: list, stds:list, save_path: str) -> None:
    plt.figure()
    plt.errorbar(alphas, means, yerr=stds)
    plt.xlabel('Alpha')
    plt.ylabel('Mean Similarity')
    plt.savefig(save_path)

def summary(sims):
    min = np.min(sims)
    print('Minimum: {:.4f}'.format(min))
    max = np.max(sims)
    print('Maximum: {:.4f}'.format(max))
    mean = np.mean(sims)
    print('Mean: {:.4f}'.format(mean))
    std = np.std(sims)
    print('Standard deviation: {:.4f}'.format(std))
    p1 = np.percentile(sims, 1)
    print('1st Percentile: {:.4f}'.format(p1))
    p2 = np.percentile(sims, 2.5)
    print('2.5th Percentile: {:.4f}'.format(p2))
    p5 = np.percentile(sims, 5)
    print('5th Percentile: {:.4f}'.format(p5))
    p25 = np.percentile(sims, 25)
    print('25th Percentile: {:.4f}'.format(p25))
    p50 = np.percentile(sims, 50)
    print('Median: {:.4f}'.format(p50))
    p75 = np.percentile(sims, 75)
    print('75th Percentile: {:.4f}'.format(p75))

    return min, max, mean, std, p1, p2, p5, p25, p50, p75

def compute_means_and_stds(model: nn.Module, loader: torch.utils.data.DataLoader,
    augmentation: str, device: torch.device, alphas: list, cosine: bool):
    for alpha in alphas:
        collect_sims(model, loader, augmentation, device, alpha, cosine)

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
    _, valid_loader = dataset_loader(args.dataset, args.batch_size,
        device)

    one_channel = True if args.dataset == 'mnist' else False
    num_classes = 1000 if args.dataset == 'imagenet' else 10

    if args.classifier:
        model = get_model(args.model, load=True, load_path=args.load_path,
            one_channel=one_channel, num_classes=num_classes,
            get_embedder=False, batchnormalize=args.batchnormalize,
            track_running_stats=False)
    else:
        model = get_model(args.model, load=True, load_path=args.load_path,
            one_channel=one_channel, num_classes=num_classes,
            get_embedder=True, batchnormalize=args.batchnormalize,
            track_running_stats=False)
    model = model.to(device)

    try:
        dim = model.get_dim()
    except RuntimeError:
        dim = num_classes

    if args.augmentation == 'none':
        model_embs, labels = get_embeddings(model, device, valid_loader, dim)

        if args.num_samples is not None:
            model_embs = model_embs[:args.num_samples,:]

        if args.cosine:
            model_embs = normalize_embeddings(model_embs)

        """
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

        model_sims = intra_class_sims + (2 * inter_class_sims)
        """

        model_sims = np.matmul(model_embs, np.transpose(model_embs))

        summary(model_sims)
        if args.save_dir is not None:
            save_path = os.path.join(args.save_dir, '{}_{}_original.png'.format(args.model, args.dataset))
            plot_sims(model_sims, args.augmentation, args.alpha, save_path)
    else:
        means = []
        stds = []
        for a in args.alpha:
            augmentation = make_augmentation(
                args.augmentation, alpha_max=args.alpha_max, device=device, random=False)
            model_sims = collect_sims(model, valid_loader, augmentation,
                device, a, args.cosine)
            print('alpha = {}'.format(a))
            _, _, mean, std, _, _, _, _, _, _ = summary(model_sims)
            means.append(mean)
            stds.append(std)
            print('\n')
            if args.save_dir is not None:
                save_path = os.path.join(args.save_dir, '{}_{}_alpha={}.png'.format(args.model, args.dataset, a))
                plot_sims(model_sims, args.augmentation, a, save_path)
        save_path = os.path.join(args.save_dir, '{}_{}_means.png'.format(args.model, args.dataset))
        plot_mean(args.alpha, means, stds, save_path)

if __name__ == '__main__':
    main()