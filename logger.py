#Inspired by the tracer from Hilbert-MLE: https://github.com/enewe101/hilbert/blob/master/hilbert/tracer.py
import os
from datetime import datetime
import yaml

config = open('../config.yaml', 'r')
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
        self.make_header(type, args)
        self.verbose = verbose

    def make_header(self, type, args):
        self.log_file.write('Experiment Time: {}\n'.format(datetime.now()))
        self.log_file.write('Type: {}\n'.format(type))
        self.log_file.write('Train Batch Size: {}\n'.format(args.train_batch_size))
        self.log_file.write('Learning Rate : {}\n'.format(args.lr))
        self.log_file.write('Seed: {}\n'.format(args.seed))
        self.log_file.write('Max Epochs: {}\n'.format(args.epochs))
        self.log_file.write('Scheduler Patience: {}\n'.format(args.patience))
        self.log_file.write('Early Stopping Patience: {}\n'.format(args.early_stop))
        self.log_file.write('Device: {}\n'.format(args.device))
        if type == 'similarity' or type == 'distillation' or type == 'linear_classifier':
           self.log_file.write('Load Path: {}\n'.format(args.load_path))
        if type == 'similarity' or type == 'distillation':
            self.log_file.write('Cosine Similarity: {}\n'.format(args.cosine))
        if type == 'similarity':
            self.log_file.write('Augmentation: {}\n'.format(args.augmentation))
            self.log_file.write('Alpha Max: {}\n'.format(args.alpha_max))
            self.log_file.write('Beta: {}\n'.format(args.beta))

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
    exp_name_start = '{}_{}_batch={}_lr={}_seed={}'.format(type, dataset,
        args.train_batch_size, args.lr, args.seed)
    if type == 'similarity' or type == 'distillation':
        exp_name_cosine = exp_name_start + '_cosine={}'.format(args.cosine)
    if type == 'similarity':
        exp_name_augmentation = exp_name_cosine + '_augmentation={}'.format(args.augmentation)
        exp_name_alphamax = exp_name_augmentation + '_alphamax={}'.format(args.alpha_max)
        exp_name_beta = exp_name_alpha_max + '_beta={}'.format(args.beta)
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

