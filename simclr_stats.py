import argparse
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from loaders import dataset_loader
from models import get_model
from data_augmentation import make_augmentation
from training import get_model_similarity

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='Batch size (default: 64)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    parser.add_argument('--load-path', type=str,
                        help='Path to trained SimCLR model.')
    parser.add_argument('--save-path', type=str,
                        help='Path to save summary statistics.')
    parser.add_argument('--cosine', action='store_true',
                         help='Use cosine similarity.')

    args = parser.parse_args()

    return args

def main():
    parser = argparse.ArgumentParser(description='SimCLR Similarity Statistics')
    args = get_args(parser)

    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    train_loader, _ = dataset_loader(args.dataset, args.batch_size, device)
    model = get_model('simclr_pretrained', load=True, load_path=args.load_path,
        get_embedder=True)
    model.to(device)
    model.eval()

    flip = transforms.RandomHorizontalFlip(p=0.5)
    grayscale = transforms.RandomGrayscale(p=0.2)

    table = np.zeros((0, 4))

    for data, _ in tqdm(train_loader):
        data = data.to(device)
        crop = make_augmentation('crop', simclr=True)
        blur = make_augmentation('blur', simclr=True)
        jitter = make_augmentation('jitter', simclr=True)
        with torch.no_grad():
            augmented_data, alpha_1 = crop.augment(data)
            augmented_data = flip(augmented_data)
            augmented_data, alpha_2 = jitter.augment(augmented_data)
            augmented_data = grayscale(augmented_data)
            augmented_data, alpha_3, _ = blur.augment(augmented_data)
            sims = get_model_similarity(model, data, augmented_data, cosine=args.cosine)

            alpha_1 = torch.reshape(alpha_1, (args.batch_size, 1)).to(device)
            alpha_2 = torch.reshape(alpha_2, (args.batch_size, 1)).to(device)
            alpha_3 = torch.reshape(alpha_3, (args.batch_size, 1)).to(device)
            sims = torch.reshape(sims, (args.batch_size, 1))
            entries = torch.cat((alpha_1, alpha_2, alpha_3, sims), dim=1).cpu().detach().numpy()
            table = np.concatenate((table, entries), axis=0)

    print('Saving similarity table...')
    np.save(args.save_path, table)
    print('Saved.')

if __name__ == '__main__':
    main()