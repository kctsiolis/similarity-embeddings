from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from embeddings import load_embeddings, save_embeddings

def tsne(embs, labels, save_path, num_classes=None, plots_path=None):
    t = TSNE()
    #Project to 2D space
    X_new = t.fit_transform(embs)
    save_embeddings(X_new, save_path)

    if plots_path is not None and num_classes is not None:
        tsne_plot(X_new, labels, num_classes, plots_path)

def tsne_plot(X_tsne, labels, num_classes, plots_path):
    if num_classes > 10:
        raise ValueError("TSNE plots limited to 10 classes.")

    plt.figure()
    colours = ['silver', 'brown', 'red', 'blue', 'orange', 'khaki', 'seagreen', 'deepskyblue', 'navy', 'fuchsia']  
    for i, colour in zip(range(num_classes), colours[:num_classes]):
        indices = np.where(labels == i)
        plt.scatter(X_tsne[indices,0], X_tsne[indices,1], c=colour, label='Class {}'.format(i))   
    plt.legend()
    plt.savefig(plots_path)