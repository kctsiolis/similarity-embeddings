"""Create a bash script with commands to run experiments.

This is useful for grid search. For hyperparameters such as learning rate
and temperature, a list can be entered. For each combination of
hyperparameters, a command is added to the bash script to train with this
hyperparameter configuration.

Experiment modes supported:
    Teacher
    Distillation
    Random
    Similarity
    Linear Classifier

Datasets supported:
    MNIST
    CIFAR-10

"""

import argparse
import os
from datetime import datetime

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--scripts-dir', type=str, default='./jobs/',
        help='Directory to store bash script containing code to run experiments.')
    parser.add_argument('--code-dir', type=str, default='..',
        help='Directory that code is located in (relative to the script directory).')
    parser.add_argument('--mode', type=str, choices=['teacher', 'distillation', 'similarity', 
        'linear_classifier', 'random'], metavar='T',
        help='The type of training.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar', 'imagenet'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--batch-size', type=int, default=[64], nargs='+', metavar='N',
        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=[0.01], nargs='+', metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default=['adam'], nargs='+',
                        help='Choice of optimizer for training.')
    parser.add_argument('--scheduler', type=str, choices=['plateau', 'cosine'], default='plateau', nargs='+',
                        help='Choice of scheduler for training.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience used in Plateau scheduler.')
    parser.add_argument('--seed', type=int, default=[42], nargs='+', metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--early-stop', type=int, default=10, metavar='E',
                        help='Number of epochs for early stopping')
    parser.add_argument('--load-path', type=str, nargs='+',
                        help='Path to the teacher model.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Name of CUDA device being used (if any). Otherwise will use CPU.')
    parser.add_argument('--cosine', action='store_true',
                        help='Use cosine similarity in the similarity and/or distillation loss (if applicable).')
    parser.add_argument('--loss', type=str, choices=['mse', 'kl'], default=['mse'], nargs='+',
                        help='Type of loss function to use for similarity experiments.')
    parser.add_argument('--temp', type=float, default=[0.01], nargs='+',
                        help='Temperature in sigmoid function converting similarity score to probability.')
    parser.add_argument('--augmentation', type=str, choices=['blur', 'color-jitter', 'random-crop'], default='blur',
                        help='Augmentation to use (if applicable).')
    parser.add_argument('--kernel-size', type=int, default=None,
                        help='Kernel size parameter for Gaussian blur.')
    parser.add_argument('--alpha-max', type=int, default=[15], nargs='+',
                        help='Largest possible augmentation strength.')
    parser.add_argument('--beta', type=float, default=[0.2], nargs='+',
                        help='Parameter of similarity probability function p(alpha).')
    parser.add_argument('--teacher-model', type=str, default='resnet18',
                        help='Model to use for the teacher (if applicable).')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model to use for the learner.')

    args = parser.parse_args()

    return args

def make_script(args):
    """Construct bash script from command line arguments."""
    script_path = os.path.join(args.scripts_dir, 'sim_exps_{}.bash'.format(datetime.now()))
    command_start = 'python {}/'.format(args.code_dir)

    with open(script_path, 'w') as f:
        command_start += 'run_training.py'
        command_start += '--mode {} '.format(args.mode)
        command_start += '--dataset {} '.format(args.dataset)
        command_start += '--epochs {} '.format(args.epochs)
        command_start += '--patience {} '.format(args.patience)
        command_start += '--early-stop {} '.format(args.early_stop)
        command_start += '--device {} '.format(args.device)
        command_start += '--model {} '.format(args.model)
        if args.cosine:
            command_start += '--cosine '

        if args.mode == 'distillation':
            command_start += '--teacher-model {} '.format(args.teacher_model)

        if args.mode == 'similarity':
            command_start += '--augmentation {} '.format(args.augmentation)

        for seed in args.seed:
            command_seed = command_start + '--seed {} '.format(seed)

            for batch_size in args.batch_size:
                command_batch = command_seed + '--batch-size {} '.format(batch_size)

                for lr in args.lr:
                    command_lr = command_batch + '--lr {} '.format(lr)

                    for optimizer in args.optimizer:
                        command_optim = command_lr + '--optimizer {} '.format(optimizer)

                        for scheduler in args.scheduler:
                            command_sched = command_optim + '--scheduler {} '.format(scheduler)

                            if args.mode == 'distillation' or args.mode == 'linear_classifier':
                                if args.load_path is not None:
                                    for load_path in args.load_path: 
                                        command_load = command_sched + '--load-path {} '.format(load_path)
                                        final_command = command_load + '\n'
                                        f.write(final_command)
                                else:
                                    f.write(command_sched + '\n')
                            elif args.mode == 'similarity':
                                for loss in args.loss:
                                    command_loss = command_sched + '--loss {} '.format(loss)
                                    for temp in args.temp:
                                        command_temp = command_loss + '--temp {} '.format(temp)
                                        for alpha_max in args.alpha_max:
                                            command_alpha = command_temp + '--alpha-max {} '.format(alpha_max)
                                            for beta in args.beta:
                                                command_beta = command_alpha + '--beta {} '.format(beta)
                                                command_kernel = command_beta + '--kernel-size {} '.format(args.kernel_size)
                                                final_command = command_kernel + '\n'
                                                f.write(final_command)
                            else:
                                final_command = command_sched + '\n'
                                f.write(final_command)       

        print('Created file {}'.format(script_path))

def main():
    parser = argparse.ArgumentParser(description='Similarity Embeddings and Distillation Experiments')
    args = get_args(parser)
    make_script(args)

if __name__ == '__main__':
    main()