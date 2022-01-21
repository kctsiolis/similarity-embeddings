"""Collection of classes and functions for neural network vision models."""

import torch
from torch import nn
import torch.nn.functional as F
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from resnet_cifar import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar

def get_model(model_str: str, load_path: str = None, 
    num_classes: int = 10) -> nn.Module:
    """Instantiate or load a specified model.
    
    Args:
        model_str: String specifying the model.
        load: Set to true to load existing model, otherwise instantiate a new one.
        load_path: Path to load model from (if we are loading).
        num_classes: Number of classes (if we are classifying).
        get_embedder: Whether or not to only get the feature embedding layers.

    Returns:
        The desired model.
    
    """ 
    if model_str == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif model_str == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif model_str == 'resnet50':
        model = resnet50(num_classes=num_classes)
    elif model_str == 'resnet101':
        model = resnet101(num_classes=num_classes)
    elif model_str == 'resnet152':
        model = resnet152(num_classes=num_classes)    
    elif model_str == 'resnet18_pretrained':
        model = resnet18(pretrained=True)
    elif model_str == 'resnet34_pretrained':
        model = resnet34(pretrained=True)
    elif model_str == 'resnet50_pretrained':
        model = resnet50(pretrained=True)    
    elif model_str == 'resnet101_pretrained':
        model = resnet101(pretrained=True)
    elif model_str == 'resnet152_pretrained':
        model = resnet152(pretrained=True)   
    elif model_str == 'resnet20_cifar':
        model = resnet20_cifar(num_classes=num_classes)
    elif model_str == 'resnet32_cifar':
        model = resnet32_cifar(num_classes=num_classes)
    elif model_str == 'resnet44_cifar':
        model = resnet44_cifar(num_classes=num_classes)
    elif model_str == 'resnet56_cifar':
        model = resnet56_cifar(num_classes=num_classes)
    else:
        raise ValueError('Model {} not defined.'.format(model_str))
        
    if load_path is not None:
        try:
            model.load_state_dict(torch.load(load_path)['model_state_dict'])
        except KeyError:
            model.load_state_dict(torch.load(load_path))

    return model