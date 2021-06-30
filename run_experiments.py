import argparse
import os
from datetime import datetime

def get_args(parser):
    parser.add_argument('--scripts-dir', type=str, default='./jobs/',
        help='Directory to store bash script containing code to run experiments.')
    parser.add_argument('--code-dir', type=str, default='..',
        help='Directory that code is located in (relative to the script directory).')
    parser.add_argument('--type', type=str, choices=['teacher', 'distillation', 'similarity', 
        'linear_classifier', 'random'], metavar='T',
        help='The type of training.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'] ,metavar='D',
        help='Dataset to train and validate on (MNIST or CIFAR).')
    parser.add_argument('--train-batch-size', type=int, default=[64], nargs='+', metavar='N',
        help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=[0.01], nargs='+', metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default=['adam'], nargs='+',
                        help='Choice of optimizer for training.')
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
    parser.add_argument('--temp', type=float, default=[1.0], nargs='+',
                        help='Temperature in sigmoid function converting similarity score to probability.')
    parser.add_argument('--augmentation', type=str, choices=['blur-sigma', 'blur-kernel'], default='blur-sigma',
                        help='Augmentation to use (if applicable).')
    parser.add_argument('--alpha-max', type=int, default=[15], nargs='+',
                        help='Largest possible augmentation strength.')
    parser.add_argument('--beta', type=float, default=[0.2], nargs='+',
                        help='Parameter of similarity probability function p(alpha).')
    parser.add_argument('--model', type=str, default='resnet18', choices=['cnn', 'resnet18', 'resnet50'],
                        help='Model to use for the learner.')

    args = parser.parse_args()

    return args

def make_script(args):
    script_path = os.path.join(args.scripts_dir, 'sim_exps_{}.bash'.format(datetime.now()))
    command_start = 'python {}/'.format(args.code_dir)

    with open(script_path, 'w') as f:
        if args.type == 'teacher':
            command_start += 'run_teacher.py '
        elif args.type == 'distillation':
            command_start += 'run_distillation.py '
        elif args.type == 'similarity':
            command_start += 'run_similarity.py '
        elif args.type == 'linear_classifier':
            command_start += 'run_linear_classifier.py '
        else:
            command_start += 'run_random.py '

        command_start += '--dataset {} '.format(args.dataset)
        command_start += '--epochs {} '.format(args.epochs)
        command_start += '--patience {} '.format(args.patience)
        command_start += '--early-stop {} '.format(args.early_stop)
        command_start += '--device {} '.format(args.device)
        command_start += '--model {} '.format(args.model)
        if args.cosine:
            command_start += '--cosine '

        if args.type == 'similarity':
            command_start += '--augmentation {} '.format(args.augmentation)

        for batch_size in args.train_batch_size:
            command_batch = command_start + '--train-batch-size {} '.format(batch_size)

            for lr in args.lr:
                command_lr = command_batch + '--lr {} '.format(lr)

                for optimizer in args.optimizer:
                    command_optim = command_lr + '--optimizer {} '.format(optimizer)

                    for seed in args.seed:
                        command_seed = command_optim + '--seed {} '.format(seed)

                        if args.type == 'distillation' or args.type == 'linear_classifier':
                            for load_path in args.load_path: 
                                command_load = command_seed + '--load-path {} '.format(load_path)
                                final_command = command_load + '\n'
                                f.write(final_command)
                        elif args.type == 'similarity':
                            for loss in args.loss:
                                command_loss = command_seed + '--loss {} '.format(loss)
                                for temp in args.temp:
                                    command_temp = command_loss + '--temp {} '.format(temp)
                                    for alpha_max in args.alpha_max:
                                        command_alpha = command_temp + '--alpha-max {} '.format(alpha_max)
                                        for beta in args.beta:
                                            command_beta = command_alpha + '--beta {} '.format(beta)
                                            final_command = command_beta + '\n'
                                            f.write(final_command)
                        else:
                            final_command = command_seed + '\n'
                            f.write(final_command)       

        print('Created file {}'.format(script_path))

def main():
    parser = argparse.ArgumentParser(description='Similarity Embeddings and Distillation Experiments')
    args = get_args(parser)
    make_script(args)

if __name__ == '__main__':
    main()