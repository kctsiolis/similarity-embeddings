'''
Evaluate many models store results in a csv. 

This file is meant to be used to evaluate students performance as teachers progressively get better and better through training 
'''

import os
import re
import argparse

import torch
import torch.nn as nn
import numpy as np

from training import predict
from models import get_model
from loaders import dataset_loader
from log_file_utils import rename_models


def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--base-path', type=str,metavar='B',help = 'Path to directory where models to evaluate are kept')
    parser.add_argument('--save-path', type=str,metavar='S',help = 'Where to save results')
    parser.add_argument('--rename',action = 'store_true',help = 'Rename models — needed if models haven\'t been renamed before')
    parser.add_argument('--evaluation-type',type = str,choices = ['teachers','students'])
    parser.add_argument('--model-type',type = str,help = 'What kind of models are stored in the base directory')    
    parser.add_argument('--dataset',type = str,choices=['mnist', 'cifar', 'cifar100', 'imagenet', 'tiny_imagenet'])
    parser.add_argument('--device', type=str, choices = ['cpu','cuda:0','cuda:1','cuda:2','cuda:3'],default = 'cuda:0')                               
 
    args = parser.parse_args()
    return args

def main():
    '''
    Evaluate many models in the given base path save results to a csv    
    '''
    parser = argparse.ArgumentParser(description='Training many models')
    args = get_args(parser)    
    
    if args.evaluation_type == 'teachers':
        # Load teacher models to evaluate                
        teacher_models = [name for name in os.listdir(args.base_path) if 'model_epoch' in name]
        epoch_numbers = [int(re.findall(r'\d+', txt)[0]) for txt in teacher_models]

        teacher_models = [os.path.abspath(os.path.join(args.base_path,teacher_path)) for teacher_path in teacher_models]
        path_list = teacher_models
    elif args.evaluation_type == 'students':
        # Load students to evaluate
        path_list = [os.path.join(args.base_path, dir) for dir in os.listdir(args.base_path)]            
        epoch_numbers = [int(re.findall(r'\d+', txt)[-1]) for txt in path_list]                        
        
        if args.rename:
            rename_models(root_path = args.base_path,path_list = path_list)
            path_list = [os.path.join(args.base_path, dir) for dir in os.listdir(args.base_path)]                                  

    
    losses,top1s,top5s = evaluate_many_models(path_list,model_type=args.model_type,device = args.device,dataset = args.dataset)
    with open(f'{args.save_path}.csv', 'w+') as csvfile:
        csvfile.write('Epoch,Top1,Top5\n')
        for epoch,top1,top5 in zip(epoch_numbers,top1s,top5s):
            csvfile.write(f'{epoch}, {top1}, {top5}\n')
        
def evaluate_many_models(path_list : list,model_type : str,device:str,dataset:str):
    '''Get model results and save for later'''
    losses = []
    top1s = []
    top5s = []
    for path in path_list:        
        loss, top1, top5 = evaluate_model(path,model_type,device,dataset)
        losses.append(loss)
        top1s.append(top1)
        top5s.append(top5)

    return losses,top1s,top5s

def evaluate_model(path : str,model_type : str,device:str,dataset : str):
    ''' Evaluate a single model'''             
    
            
    if os.path.isdir(path):
        model_path = os.path.join(path,'model.pt')
    else:
        model_path = path

    # Set some assumed parameters
    seed = 1    
    model = model_type
    batch_size = 128
    train_set_fraction = 1
    validate = True if ((dataset == 'imagenet') or (dataset == 'tiny_imagenet')) else False    
    precision = '32'
    split = 'test'
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(device)

    loader1, loader2 = dataset_loader(
        dataset, batch_size, 
        train_set_fraction=train_set_fraction, 
        validate=validate)
                
    one_channel = dataset == 'mnist'
    if dataset == 'imagenet':
        num_classes = 1000
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny_imagenet': 
        num_classes = 200
    else:
        num_classes = 10

        
    model = get_model(
        model, load=True, load_path=model_path,
        one_channel=one_channel, num_classes=num_classes)            
    model.to(device)

    

    if split == 'train':
        metrics = predict(model, device, loader1, nn.CrossEntropyLoss(),precision)
    elif split == 'val':
        metrics = predict(model, device, loader2, nn.CrossEntropyLoss(),precision)
    else:
        metrics = predict(model, device, loader2, nn.CrossEntropyLoss(),precision)

    print('Loss: {:.6f}'.format(metrics[0]))
    print('Top-1 Accuracy: {:.2f}'.format(metrics[1]))
    print('Top-5 Accuracy: {:.2f}'.format(metrics[2]))


    return metrics[0],metrics[1], metrics[2] 






if __name__ == '__main__':
    main()







