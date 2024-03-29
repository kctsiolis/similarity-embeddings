"""Evaluate a model on a classification task.

Models supported:
    ResNet18
    ResNet50
    ResNet152

Datasets supported:
    MNIST
    CIFAR-10
    ImageNet

"""

import argparse
from torch import nn
from run_base import get_base_args, run_base
from models.models import get_model
from training.training import predict


# def get_args(parser):
#     """Collect command line arguments."""
#     parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10','cifar100', 'imagenet'] ,metavar='D',
#         help='Dataset to train and validate on (MNIST or CIFAR).')
    
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#         help='Batch size (default: 64)')
#     parser.add_argument('--train-set-fraction', type=float, default=1.0,
#                         help='Fraction of training set to train on.')
#     parser.add_argument('--validate', action='store_true',
#                         help='Evaluate on a held out validation set (as opposed to the test set).')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--device', type=str, default='cpu',
#                         help='Name of CUDA device being used (if any).')
#     parser.add_argument('--model', type=str, default='resnet18',
#                         help='Choice of model.')
#     parser.add_argument('--load-path', type=str,
#                         help='Path to the teacher model.')                                                                      

#     args = parser.parse_args()

#     return args    

def get_model_args(parser):
    parser.add_argument('--model', type=str, help='Kind of model.')
    parser.add_argument('--model-path', type=str, help='Path to the model')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'] ,metavar='D',
        help='Choice of either training, validation, or test subset (where applicable).')

    return parser

def get_args(parser):
    parser = get_base_args(parser)
    parser = get_model_args(parser)

    args = parser.parse_args()
    return args

def main():
    parser = argparse.ArgumentParser(description='Evaluating a model')
    args = get_args(parser)

    train_loader, val_loader, num_classes, device = run_base(args)
    ## BASE GETS LOADER AND DEVICE
    
    model = get_model(args.model, args.model_path, num_classes=num_classes)
    model.end_to_end_mode()
    model.to(device)    
    
    if args.split == 'train':
        metrics = predict(model, device, train_loader, nn.CrossEntropyLoss())
    elif args.split == 'val':
        metrics = predict(model, device, val_loader, nn.CrossEntropyLoss())
    elif args.split == 'test':
        metrics = predict(model, device, val_loader, nn.CrossEntropyLoss())
    else:
        raise ValueError('Must specify the dataset split to use')

    print('Loss: {:.6f}'.format(metrics[0]))
    print('Top-1 Accuracy: {:.2f}'.format(metrics[1]))
    print('Top-5 Accuracy: {:.2f}'.format(metrics[2]))



# def main():
#     """Load arguments, the dataset, and initiate the training loop."""
#     parser = argparse.ArgumentParser(description='Training the teacher model')
#     args = get_args(parser)

#     #Set random seed
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True

#     device = torch.device(args.device)

#     loader1, loader2 = dataset_loader(
#         args.dataset, args.batch_size, 
#         train_set_fraction=args.train_set_fraction, 
#         validate=args.validate)
            
#     one_channel = args.dataset == 'mnist'
#     num_classes = 1000 if args.dataset == 'imagenet' else 10

#     model = get_model(
#         args.model, load=True, load_path=args.load_path, 
#         one_channel=one_channel, num_classes=num_classes)            
#     model.to(device)

#     if args.split == 'train':
#         metrics = predict(model, device, loader1, nn.CrossEntropyLoss())
#     elif args.split == 'val':
#         metrics = predict(model, device, loader2, nn.CrossEntropyLoss())
#     else:
#         raise ValueError('Must specify which dataset to use')

#     print('Loss: {:.6f}'.format(metrics[0]))
#     print('Top-1 Accuracy: {:.2f}'.format(metrics[1]))
#     print('Top-5 Accuracy: {:.2f}'.format(metrics[2]))

   
if __name__ == '__main__':
    main()