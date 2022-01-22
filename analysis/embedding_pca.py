import argparse
import numpy as np
from sklearn.decomposition import PCA
from analysis.embeddings import load_embeddings, normalize_embeddings

def get_args(parser):
    parser.add_argument('-n', dest='num_components', type=int, help='Number of principal components.',
        default=5)
    parser.add_argument('--embs', dest='embs', help='.npy file containing embeddings.',
        default='/mnt/data/scratch/konstantinos.tsiolis/win5_embs/W.npy')
    parser.add_argument('--normalize', dest='normalize', choices=['true', 'false'], default='false',
        help='Whether or not to L2 normalize the rows of the matrices.')
    parser.add_argument('--save_path', dest='save_path', default='None',
        help='Path to location where reduced embeddings will be saved (optional).')

    return parser

def embedding_pca(args):
    embs_file = args.embs
    normalize = args.normalize
    num_components = args.num_components
    save_path = args.save_path

    print("Loading embeddings...")
    W = load_embeddings(embs_file)
    print("Embeddings loaded.")

    if args.normalize == 'true':
        print("Embeddings will be L2 normalized.")
        W = normalize_embeddings(W)

    print("Applying PCA with {} components...".format(num_components))
    pca = PCA(n_components = num_components)
    W = pca.fit_transform(W)

    print("Explained variance per component:")
    print(pca.explained_variance_ratio_)

    print("Total explained variance:")
    print(np.sum(pca.explained_variance_ratio_))

    if save_path != None:
        print("Saving reduced embeddings...")
        np.save(save_path, W)
        print("Saved.")

    return W

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()
    embedding_pca(args)