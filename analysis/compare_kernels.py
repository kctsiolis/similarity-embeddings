#Given two matrices M1 and M2, computes ||M1^T * M1 - M2^T * M2||

import argparse
import numpy as np
from analysis.embeddings import load_embeddings, normalize_embeddings, compute_similarity_matrix

def get_args(parser):
    parser.add_argument('--m1', dest='m1', help='.npy file containing first matrix.',
        default='/mnt/data/scratch/konstantinos.tsiolis/win5_embs/W.npy')
    parser.add_argument('--m2', dest='m2', help='.npy file containing second matrix.',
        default='/mnt/data/scratch/konstantinos.tsiolis/win5_embs/V.npy')
    parser.add_argument('--metric', dest='metric', choices=['frobenius', 'mean', 'agreement'], default='frobenius',
        help='Distance to use between Gram matrices.')
    parser.add_argument('--thresh', dest='threshold', type=float, default=0.5,
        help='Threshold for similarity / dissimilarity.')
    parser.add_argument('--normalize', dest='normalize', choices=['true', 'false'], default='false',
        help='Whether or not to L2 normalize the rows of the matrices.')

    return parser

def compare_kernels(args):
    #Get filenames for matrices
    m1_file = args.m1
    m2_file = args.m2

    normalize = args.normalize

    print("Loading first matrix...")
    m1 = load_embeddings(m1_file)
    print("Loaded first matrix.")
    print("Loading second matrix...")
    m2 = load_embeddings(m2_file)
    print("Loaded second matrix.\n")

    if normalize == 'true':
        print("Matrices will be L2 normalized.")
        m1 = normalize_embeddings(m1)
        m2 = normalize_embeddings(m2)
    
    print("Computing first kernel...")
    k1 = compute_similarity_matrix(m1)
    print("Computed first kernel.")
    print("Computing second kernel...")
    k2 = compute_similarity_matrix(m2)
    print("Computed second kernel.\n")

    dist = compute_distance(k1,k2,args)
    print("Distance = {}".format(dist))

def compute_distance(k1, k2, args):
    metric = args.metric

    if metric == 'frobenius':
        dist = frobenius_distance(k1,k2)
    elif metric == 'mean':
        dist = mean_distance(k1,k2)
    else:
        threshold = args.threshold
        dist = agreement_rate(k1,k2,threshold)

    print("The {} distance will be used.\n".format(metric))

    return dist

def frobenius_distance(k1, k2):
    #Compute the difference between the Gram matrices first
    diff = k1 - k2
    dist = np.linalg.norm(diff)

    return dist

#Gives the average entrywise difference
def mean_distance(k1, k2):
    diff = np.absolute(k1 - k2)
    dist = np.mean(diff)

    return dist

#Gives percentage of entries which agree on 0-1 similarity
def agreement_rate(k1, k2, threshold):
    print("The threshold is {:.2f}.\n".format(threshold))
    k1[k1 >= threshold] = 1
    k1[k1 < threshold] = 0
    k2[k2 >= threshold] = 1
    k2[k2 < threshold] = 0

    return np.mean(k1 == k2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()
    compare_kernels(args)