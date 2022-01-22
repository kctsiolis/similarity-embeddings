import numpy as np
import torch 

def load_embeddings(path):
    return np.load(path)

def save_embeddings(embs, path):
    np.save(path, embs)

#Given a matrix M, compute M * M^T
def compute_similarity_matrix(m):
    return np.matmul(m,np.transpose(m))

def normalize_embeddings(m):
    l2norm = np.sqrt((m * m).sum(axis=1))
    return m / l2norm.reshape(m.shape[0],1)