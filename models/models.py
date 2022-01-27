"""Collection of classes and functions for neural network vision models."""

import torch
from torch import nn
import torch.nn.functional as F
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.resnet_cifar import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar
from models.general import CLIPDistill


def get_model(model_str: str, load_path: str = None,
              num_classes: int = 10, project: bool = False, projection_dim: int = None) -> nn.Module:
    if model_str == 'resnet18':
        model = resnet18(num_classes, project, projection_dim)
    elif model_str == 'resnet34':
        model = resnet34(num_classes, project, projection_dim)
    elif model_str == 'resnet50':
        model = resnet50(num_classes, project, projection_dim)
    elif model_str == 'resnet101':
        model = resnet101(num_classes, project, projection_dim)
    elif model_str == 'resnet152':
        model = resnet152(num_classes, project, projection_dim)
    elif model_str == 'resnet18_pretrained':
        model = resnet18(1000, pretrained=True)
    elif model_str == 'resnet34_pretrained':
        model = resnet34(1000, pretrained=True)
    elif model_str == 'resnet50_pretrained':
        model = resnet50(1000, pretrained=True)
    elif model_str == 'resnet101_pretrained':
        model = resnet101(1000, pretrained=True)
    elif model_str == 'resnet152_pretrained':
        model = resnet152(1000, pretrained=True)
    elif model_str == 'resnet20_cifar':
        model = resnet20_cifar(num_classes, project, projection_dim)
    elif model_str == 'resnet32_cifar':
        model = resnet32_cifar(num_classes, project, projection_dim)
    elif model_str == 'resnet44_cifar':
        model = resnet44_cifar(num_classes, project, projection_dim)
    elif model_str == 'resnet56_cifar':
        model = resnet56_cifar(num_classes, project, projection_dim)
    elif model_str == 'clip_distill':
        model = resnet56_cifar(num_classes, project, projection_dim)
    else:
        raise ValueError('Model {} not defined.'.format(model_str))

    if load_path is not None:
        try:
            model.load_state_dict(torch.load(load_path)['model_state_dict'])
        except KeyError:
            model.load_state_dict(torch.load(load_path))

    return model


def get_model_embedding_dim(model_str: str) -> int:
    if model_str == 'resnet18':
        dim = 512
    elif model_str == 'resnet34':
        dim = 512
    elif model_str == 'resnet50':
        dim = 2048
    elif model_str == 'resnet101':
        dim = 2048
    elif model_str == 'resnet152':
        dim = 2048
    elif model_str == 'resnet18_pretrained':
        dim = 512
    elif model_str == 'resnet34_pretrained':
        dim = 512
    elif model_str == 'resnet50_pretrained':
        dim = 2048
    elif model_str == 'resnet101_pretrained':
        dim = 2048
    elif model_str == 'resnet152_pretrained':
        dim = 2048
    elif model_str == 'resnet20_cifar':
        dim = 512
    elif model_str == 'resnet32_cifar':
        dim = 64
    elif model_str == 'resnet44_cifar':
        dim = 512
    elif model_str == 'resnet56_cifar':
        dim = 64
    else:
        raise ValueError('Model {} not defined.'.format(model_str))

    return dim
