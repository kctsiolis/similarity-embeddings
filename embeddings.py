import numpy as np 

def load_embeddings(path):
    return np.load(path)

def save_embeddings(embs, path):
    np.save(path, embs)
