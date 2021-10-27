"""Class and functions for logging training progress and results.

Inspired by the tracer from Hilbert-MLE: https://github.com/enewe101/hilbert/blob/master/hilbert/tracer.py

Attributes:
    LOGS_DIR: Directory where logs are stored

"""

import os
from datetime import datetime
from argparse import Namespace
import yaml

try:
    config = open('../config.yaml', 'r')
except FileNotFoundError:
    config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)
LOGS_DIR = parsed_config['logs_dir']

class Logger:
    """Training status and results tracker.

    Attributes:
        dir (str): Path to directory where log is located.
        log_path (str): Path to log.txt file in log directory.
        results_path (str): Path to results.txt file in log directory.
        model_path (str): Path to saved model in log directory.
        plots_dir (str): Path to directory where plots are located.
        log_file (TextIO): File object of the log file.
        results_file (TextIO): File object of the results file.
        verbose (bool): Whether or not to print log info to stdout. 

    """

    def __init__(self, args: Namespace, 
        save: bool = True, verbose: bool = True):
        """Instantiate logger object.

        Args:
            args: Command line arguments used to run experiment.
            verbose: Whether or not to print logger info to stdout.

        """
        
        self.verbose = verbose
        if save:
            self.dir = make_log_dir(args)
            os.mkdir(self.dir)
            self.log_path = os.path.join(self.dir, 'log.txt')
            self.results_path = os.path.join(self.dir, 'results.txt')
            self.model_path = os.path.join(self.dir, 'model.pt')
            self.plots_dir = os.path.join(self.dir, 'plots')
            os.mkdir(self.plots_dir)
            self.log_file = open(self.log_path, 'w')
            self.results_file = open(self.results_path, 'w')
            self.make_header(args)
        else:
            self.log_file = None
            self.results_file = None
            self.model_path = None   


            # Noah's hacky solution
            self.dir = make_log_dir(args)
            os.mkdir(self.dir)
            self.log_path = os.path.join(self.dir, 'log.txt')
            self.results_path = os.path.join(self.dir, 'results.txt')
            self.model_path = os.path.join(self.dir, 'model.pt')
            self.plots_dir = os.path.join(self.dir, 'plots')
            os.mkdir(self.plots_dir)
            self.log_file = open(self.log_path, 'w')
            self.results_file = open(self.results_path, 'w')
            # self.make_header(args)         

    def make_header(self, args: Namespace) -> None:
        """Start the log with a header giving general experiment info.

        Args:
            args: Command line arguments used when running experiment.

        """
        self.log('Experiment Time: {}'.format(datetime.now()))
        self.log('Mode: {}'.format(args.mode))
        self.log('Dataset: {}'.format(args.dataset))
        if args.mode == 'distillation':
            self.log('Distillation Type: {}'.format(args.distillation_type))
            if args.distillation_type == 'class_probs':
                self.log('c: {}'.format(args.c))
        self.log('Batch Size: {}'.format(args.batch_size))
        self.log('Learning Rate : {}'.format(args.lr))
        self.log('Optimizer: {}'.format(args.optimizer))
        self.log('Scheduler: {}'.format(args.scheduler))
        self.log('Learning Rate Warmup Iters: {}'.format(args.lr_warmup_iters))
        self.log('Training Set Fraction: {}'.format(args.train_set_fraction))
        self.log('Validate: {}'.format(args.validate))
        self.log('Seed: {}'.format(args.seed))
        self.log('Max Epochs: {}'.format(args.epochs))
        if args.scheduler == 'plateau':
            self.log('Scheduler Patience: {}'.format(args.plateau_patience))
        self.log('Early Stopping Patience: {}'.format(args.early_stop))
        self.log('Device: {}'.format(args.device))
        self.log('Model: {}'.format(args.student_model))
        if args.mode == 'distillation':
            self.log('Teacher Model: {}'.format(args.teacher_model))
        if args.teacher_path is not None:
           self.log('Teacher Path: {}'.format(args.teacher_path))
        if args.student_path is not None:
            self.log('Student Path: {}'.format(args.student_path))
        if args.mode == 'similarity' or args.mode == 'distillation':
            self.log('Cosine Similarity: {}'.format(args.cosine))
        if args.mode == 'similarity':
            if args.loss == 'kl':
                self.log('Temp: {}'.format(args.temp))
            self.log('Augmentation: {}'.format(args.augmentation))
            self.log('Alpha Max: {}'.format(args.alpha_max))
            self.log('Beta: {}'.format(args.beta))

    def log(self, string: str) -> None:
        """Write a string to the log.

        Args:
            string: String to write.

        """
        if self.log_file is not None:
            self.log_file.write(string + '\n')
        if self.verbose:
            print(string)

    def log_results(self, string: str) -> None:
        """Write results to the results file.
        
        Args:
            string: String to write to results file.

        """
        if self.results_file is not None:
            self.results_file.write(string + '\n')
        if self.verbose:
            print(string)

    def get_model_path(self):
        return self.model_path

    def get_plots_dir(self):
        return self.plots_dir

def make_log_dir(args: Namespace) -> None:
    """Create directory to store log, results file, model, and plots.

    Args:
        args: Command line arguments used to run experiment.

    """

    exp_name_start = '{}_{}_batch={}_lr={}_optim={}_seed={}_model={}'.format(
        args.mode, args.dataset, args.batch_size, args.lr, args.optimizer, args.seed, args.student_model)
    if args.mode == 'similarity' or args.mode == 'distillation':
        exp_name_cosine = exp_name_start + '_cosine={}'.format(args.cosine)
    if args.mode == 'similarity':
        exp_name_loss = exp_name_cosine + '_loss={}'.format(args.loss)
        exp_name_augmentation = exp_name_loss + '_augmentation={}'.format(args.augmentation)
        exp_name_alphamax = exp_name_augmentation + '_alphamax={}'.format(args.alpha_max)
        exp_name_beta = exp_name_alphamax + '_beta={}'.format(args.beta)
        if args.loss == 'kl':
            exp_name_temp = exp_name_beta + '_temp={}'.format(args.temp)
            dir = os.path.join(LOGS_DIR, exp_name_temp)
        else:
            dir = os.path.join(LOGS_DIR, exp_name_beta)
    elif args.mode == 'distillation':
        dir = os.path.join(LOGS_DIR, exp_name_cosine)
    else:
        dir = os.path.join(LOGS_DIR, exp_name_start)

    if os.path.exists(dir):
        exists = True
        i = 1
        while exists:
            new_dir = dir + '_{}'.format(i)
            exists = os.path.exists(new_dir)
            i += 1
        return new_dir
    else:
        return dir

