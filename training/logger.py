"""Class and functions for logging training progress and results.

Inspired by the tracer from Hilbert-MLE: https://github.com/enewe101/hilbert/blob/master/hilbert/tracer.py

Attributes:
    LOGS_DIR: Directory where logs are stored

"""

import os
from datetime import datetime
from argparse import Namespace
import yaml

config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)
LOGS_DIR = parsed_config['logs_dir']

if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)

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

    def __init__(self, mode: str, args: Namespace, verbose: bool = True):
        """Instantiate logger object.

        Args:
            mode: Training mode.
            args: Command line arguments used to run experiment.
            verbose: Whether or not to print logger info to stdout.

        """
        self.mode = mode
        if mode == 'teacher':
            self.model_str = args.teacher_model
        elif mode == 'distillation':
            self.model_str = args.student_model
        elif mode == 'linear_classifier':
            self.model_str = args.model
        elif mode == 'clip_distillation':
            self.model_str = args.student_model
        self.verbose = verbose
        self.save = not args.no_save
        if self.save:
            self.dir = make_log_dir(mode, args)
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
            self.plots_dir = None       

    def make_header(self, args: Namespace) -> None:
        """Start the log with a header giving general experiment info.

        Args:
            args: Command line arguments used when running experiment.

        """
        self.log('Experiment Time: {}'.format(datetime.now()))
        self.log('Mode: {}'.format(self.mode))
        self.log('Dataset: {}'.format(args.dataset))
        if self.mode == 'clip_distill':
            self.log('CLIP Distillation')            
        self.log('Batch Size: {}'.format(args.batch_size))
        self.log('Learning Rate : {}'.format(args.lr))
        self.log('Optimizer: {}'.format(args.optimizer))
        self.log('Scheduler: {}'.format(args.scheduler))
        self.log('Learning Rate Warmup Iters: {}'.format(args.lr_warmup_iters))
        self.log('Training Set Fraction: {}'.format(args.train_set_fraction))
        self.log('Seed: {}'.format(args.seed))
        self.log('Max Epochs: {}'.format(args.epochs))
        if args.scheduler == 'plateau':
            self.log('Scheduler Patience: {}'.format(args.plateau_patience))
        self.log('Early Stopping Patience: {}'.format(args.early_stop))
        self.log('Model: {}'.format(self.model_str))
        if self.mode == 'distillation':
            self.log('Teacher Model: {}'.format(args.teacher_model))
            self.log('Teacher Path: {}'.format(args.teacher_path))
            self.log('Student Path: {}'.format(args.student_path))   
            self.log('Margin: {}'.format(args.margin))
            if args.margin:
                self.log('Margin Value: {}'.format(args.margin_value))  

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

def make_log_dir(model_str, args: Namespace) -> None:
    """Create directory to store log, results file, model, and plots.

    Args:
        args: Command line arguments used to run experiment.

    """
    exp_name_start = '{}_{}_batch={}_lr={}_optim={}_seed={}_model={}'.format(
        model_str, args.dataset, args.batch_size, args.lr, args.optimizer, args.seed, model_str)
    
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