#Inspired by the tracer from Hilbert-MLE: https://github.com/enewe101/hilbert/blob/master/hilbert/tracer.py
import os
from datetime import datetime
import yaml

try:
    config = open('../config.yaml', 'r')
except FileNotFoundError:
    config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)
LOGS_DIR = parsed_config['logs_dir']

class Logger:

    def __init__(self, type, dataset, args, verbose=True):
        self.dir = make_log_dir(type, dataset, args)
        os.mkdir(self.dir)
        self.log_path = os.path.join(self.dir, 'log.txt')
        self.results_path = os.path.join(self.dir, 'results.txt')
        self.model_path = os.path.join(self.dir, 'model.pt')
        self.plots_dir = os.path.join(self.dir, 'plots')
        os.mkdir(self.plots_dir)
        self.log_file = open(self.log_path, 'w')
        self.results_file = open(self.results_path, 'w')
        self.verbose = verbose
        self.make_header(type, args)

    def make_header(self, type, args):
        self.log('Experiment Time: {}\n'.format(datetime.now()))
        self.log('Type: {}\n'.format(type))
        if type == 'teacher' or type == 'linear_classifier' or type == 'random':
            self.log('Loss Function: Cross Entropy')
        elif type == 'distillation':
            self.log('Loss Function: MSE')
        else:
            self.log('Loss Function: {}'.format(args.loss))
        self.log('Train Batch Size: {}'.format(args.train_batch_size))
        self.log('Learning Rate : {}'.format(args.lr))
        self.log('Optimizer: {}'.format(args.optimizer))
        self.log('Scheduler: {}'.format(args.scheduler))
        self.log('Seed: {}'.format(args.seed))
        self.log('Max Epochs: {}'.format(args.epochs))
        self.log('Scheduler Patience: {}'.format(args.patience))
        self.log('Early Stopping Patience: {}'.format(args.early_stop))
        self.log('Device: {}'.format(args.device))
        self.log('Model: {}'.format(args.model))
        if type == 'distillation' or type == 'linear_classifier':
           self.log('Load Path: {}'.format(args.load_path))
        if type == 'similarity' or type == 'distillation':
            self.log('Cosine Similarity: {}'.format(args.cosine))
        if type == 'similarity':
            if args.loss == 'kl':
                self.log('Temp: {}'.format(args.temp))
            self.log('Augmentation: {}'.format(args.augmentation))
            self.log('Alpha Max: {}'.format(args.alpha_max))
            self.log('Beta: {}'.format(args.beta))

    def log(self, string):
        self.log_file.write(string + '\n')
        if self.verbose:
            print(string)

    def log_results(self, string):
        self.results_file.write(string + '\n')
        if self.verbose:
            print(string)

    def get_model_path(self):
        return self.model_path

    def get_plots_dir(self):
        return self.plots_dir

def make_log_dir(type, dataset, args):
    exp_name_start = '{}_{}_batch={}_lr={}_optim={}_seed={}_model={}'.format(type, dataset,
        args.train_batch_size, args.lr, args.optimizer, args.seed, args.model)
    if type == 'similarity' or type == 'distillation':
        exp_name_cosine = exp_name_start + '_cosine={}'.format(args.cosine)
    if type == 'similarity':
        exp_name_loss = exp_name_cosine + '_loss={}'.format(args.loss)
        exp_name_augmentation = exp_name_loss + '_augmentation={}'.format(args.augmentation)
        exp_name_alphamax = exp_name_augmentation + '_alphamax={}'.format(args.alpha_max)
        exp_name_beta = exp_name_alphamax + '_beta={}'.format(args.beta)
        if args.loss == 'kl':
            exp_name_temp = exp_name_beta + '_temp={}'.format(args.temp)
            dir = os.path.join(LOGS_DIR, exp_name_temp)
        else:
            dir = os.path.join(LOGS_DIR, exp_name_beta)
    elif type == 'distillation':
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

