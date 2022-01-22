''' Get embeddings and perform svd for many models '''

import os
import re
import argparse

import torch
import torch.nn as nn
import numpy as np
import yaml


from training.training import predict
from models.models import get_model
from training.loaders import dataset_loader
from log_file_utils import rename_models
from training.training import get_embeddings

try:
    config = open('../config.yaml', 'r')
except FileNotFoundError:
    config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--save-dir', type=str,metavar='B',help = 'Path to directory where models to evaluate are kept')    
    parser.add_argument('--model-path', type=str,metavar='C')        
    parser.add_argument('--model-type',type = str,help = 'What kind of models are stored in the base directory')   
    parser.add_argument('--get-embedder',action = 'store_true',help = 'Do we need to get embedder from models?')    
    parser.add_argument('--train-set',action = 'store_true',help = 'Get train set embeddings instead?')    
    parser.add_argument('--dataset',type = str,choices=['mnist', 'cifar', 'cifar100', 'imagenet', 'tiny_imagenet'])
    parser.add_argument('--device', type=str, choices = ['cpu','cuda:0','cuda:1','cuda:2','cuda:3'],default = 'cuda:0')                               
    parser.add_argument('--emb-dim', type=int)                               
 
    args = parser.parse_args()
    return args

def main():
    '''
    Load models and get embeddings for them
    '''
    parser = argparse.ArgumentParser(description='Training many models')
    args = get_args(parser)    
                
    get_model_embeddings(args.model_path,args.save_dir,args.model_type,args.get_embedder,args.device,args.dataset,args.emb_dim,args.train_set)    



def get_model_embeddings(model_path : str,save_dir : str,model_type : str,get_embedder : bool ,device:str,dataset:str,emb_dim : int, train_set : bool):
    '''Get model embeddings for many models and save for later'''
    

    # Load the dataset
    validate = (dataset == 'imagenet') or (dataset == 'tiny_imagenet')    

    loader1, loader2 = dataset_loader(
        dataset, batch_size = 64, 
        train_set_fraction=1, 
        validate=validate)
    
    if train_set:
        loader2 = loader1

    ## Load the models 
    one_channel = (dataset == 'mnist')    
    if dataset == 'imagenet':
        num_classes = 1000
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny_imagenet': 
        num_classes = 200
    else:
        num_classes = 10
            
    print(f'Getting embeddings')        
    model = get_model(
        model_type, load=True, load_path=model_path,
        one_channel=one_channel, num_classes=num_classes,get_embedder=get_embedder)            
    
    embeddings,labels = get_embeddings(model,device,loader2,emb_dim)            
                
    # We'll save all the embeddings to a directory  
    print(f'Saving embeddings...')                  
    embed_path = os.path.join(save_dir,'embeddings.npy')
    label_path = os.path.join(save_dir,'labels.npy')
    os.makedirs(save_dir)        
    np.save(embed_path,embeddings)
    np.save(label_path,labels)    





if __name__ == '__main__':
    main()







