"""Code for training loops and loss computation.

Supports supervised training, similarity-based embedding training,
and similarity-based distillation training.

Includes code for training summary and plots.

Loosely based on code from PyTorch example
https://github.com/pytorch/examples/blob/master/mnist/main.py

"""
from argparse import Namespace
import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import numpy as np                                    
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from training.logger import Logger
from training.data_augmentation import SimCLRTransform
import time

from training.distillers import SimilarityDistiller, KD, WeightedDistiller

class Trainer():

    def __init__(
            self, model: nn.Module, train_loader: DataLoader, 
            val_loader: DataLoader, device: torch.device, logger: Logger,
            args: Namespace):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            self.eval_set = 'Test'
        else:
            self.eval_set = 'Validation'
        self.device = device
        self.epochs = args.epochs
        self.itr = 0
        self.lr = args.lr    
        self.lr_warmup_iters = args.lr_warmup_iters  

        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError('Only Adam and SGD optimizers are supported.')

        self.sched_str = args.scheduler
        if self.sched_str == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=args.plateau_patience)
            self.plateau_factor = 0.1
            self.change_epochs = []
        elif self.sched_str == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200)
            self.change_epochs = None
        elif self.sched_str == 'exponential':
            self.scheduler = LambdaLR(self.optimizer, lr_lambda = lambda x :1e-5 + (1e-2 - 1e-5) * x * 195 ,verbose = True)
            self.change_epochs = None

        self.early_stop = args.early_stop
        self.log_interval = args.log_interval
        self.save_each_epoch = args.save_each_epoch
        if args.plot_interval is None:
            self.plot_interval = len(self.train_loader) // 10
        else:
            self.plot_interval = args.plot_interval
        self.logger = logger
        self.model_path = logger.get_model_path()        
        self.plots_dir = logger.get_plots_dir()

        self.iters = [] 
        self.logged_train_losses = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_accs5 = []
        self.val_accs5 = []

    def lr_warmup_factor(self):
        return min(float(self.itr + 1) / max(self.lr_warmup_iters, 1), 1.0)

    def update_lr(self):
        if self.itr < self.lr_warmup_iters:
            self.lr = self.lr * self.lr_warmup_factor()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def train(self):
        epochs_until_stop = self.early_stop
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()        
            train_loss, train_acc, train_acc5 = self.train_epoch(epoch)
            epoch_end = time.time()
            self.logger.log('Epoch Duration: {:.2f} s'.format(epoch_end-epoch_start))
            val_loss, val_acc, val_acc5 = self.validate()

            log_str = '\nTraining set: Average loss: {:.6f}\n'.format(
            train_loss)
            if train_acc is not None:
                log_str += 'Training set: Average top-1 accuracy: {:.2f}%\n'.format(train_acc)
                log_str += 'Training set: Average top-5 accuracy: {:.2f}%\n'.format(train_acc5)
            log_str += '\n{} set: Average loss: {:.6f}\n'.format(self.eval_set, val_loss)
            if val_acc is not None:
                log_str += '{} set: Top-1 Accuracy: {:.2f}%\n'.format(self.eval_set, val_acc)
                log_str += '{} set: Top-5 Accuracy: {:.2f}%\n'.format(self.eval_set, val_acc5)
            
            self.logger.log(log_str)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.train_accs5.append(train_acc5)
            self.val_accs5.append(val_acc5)

            if self.save_each_epoch:
                name, ext = os.path.splitext(self.model_path)                        
                current_model_path = f'{name}_epoch{epoch}{ext}'
                self.save_model(current_model_path, epoch)

            #Check if validation loss is worsening
            if val_loss > min(self.val_losses):
                epochs_until_stop -= 1
                if epochs_until_stop == 0: #Early stopping initiated
                    break
            else:
                epochs_until_stop = self.early_stop

                #Save the current model
                if self.logger.save:
                    if self.model_path is not None:
                        current_model_path = self.model_path
                        self.save_model(current_model_path, epoch)
                        if self.logger.mode == 'clip_distillation':
                            student_path = current_model_path.replace('model.pt','student.pt')
                            torch.save(self.model.student.state_dict(),student_path)                                                        


            if self.sched_str == 'plateau':
                self.scheduler.step(val_loss)
                if self.optimizer.param_groups[0]['lr'] == self.plateau_factor * self.lr:
                    self.change_epochs.append(epoch)
                    lr = self.plateau_factor * self.lr
                    self.logger.log("Learning rate decreasing to {}\n".format(lr))
            else:
                self.scheduler.step()

        self.train_report()

    def train_epoch(self, epoch):
        pass

    def validate(self):
        pass

    def train_report(self):
        best_epoch = np.argmin(self.val_losses)
        self.logger.log("Training complete.\n")
        self.logger.log_results("Best Epoch: {}".format(best_epoch + 1))
        self.logger.log_results("Training Loss: {:.6f}".format(
            self.train_losses[best_epoch]))
        if self.train_accs[best_epoch] is not None:
            self.logger.log_results("Training Top-1 Accuracy: {:.2f}".format(
                self.train_accs[best_epoch]))
            self.logger.log_results("Training Top-5 Accuracy: {:.2f}".format(
                self.train_accs5[best_epoch]))
        self.logger.log_results("{} Loss: {:.6f}".format(
            self.eval_set, self.val_losses[best_epoch]))
        if self.val_accs[best_epoch] is not None:
            self.logger.log_results("{} Top-1 Accuracy: {:.2f}".format(
                self.eval_set, self.val_accs[best_epoch]))
            self.logger.log_results("{} Top-5 Accuracy: {:.2f}".format(
                self.eval_set, self.val_accs5[best_epoch]))

        #Save loss and accuracy plots
        if self.logger.save:
            train_val_plots(
                self.train_losses, self.val_losses, "Loss", self.plots_dir, self.change_epochs)
            if self.train_accs is not None and self.val_accs is not None:
                train_val_plots(
                    self.train_accs, self.val_accs, "Accuracy", self.plots_dir, self.change_epochs)

    def save_model(self, path, epoch):
        self.logger.log('Saving model...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        self.logger.log('Model saved.')

class SupervisedTrainer(Trainer):
    def __init__(
            self, model: nn.Module, train_loader: DataLoader, 
            val_loader: DataLoader, device: torch.device, logger: Logger,
            args: Namespace):
        super().__init__(
            model, train_loader, val_loader, device, logger, args)
        self.loss_function = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = AverageMeter()
        train_top1_acc = AverageMeter()
        train_top5_acc = AverageMeter()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.update_lr()
            data, target = data.to(self.device, non_blocking = True), target.to(self.device, non_blocking = True)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(data)
            loss = self.loss_function(output, target)
            top1_acc, top5_acc = compute_accuracy(output, target)
            train_loss.update(loss.item())
            train_top1_acc.update(top1_acc)
            train_top5_acc.update(top5_acc)
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % self.log_interval == 0:
                logged_loss = train_loss.get_avg()
                self.iters.append((epoch-1) + batch_idx / len(self.train_loader))
                self.logged_train_losses.append(logged_loss)
                log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                    epoch, batch_idx * len(data), int(len(self.train_loader.dataset)),
                    100. * batch_idx / len(self.train_loader), logged_loss)
                self.logger.log(log_str)
            if batch_idx % self.plot_interval == 0 and self.logger.save:
                train_loss_plot(
                    self.iters, self.logged_train_losses, self.plots_dir)
            self.itr += 1           

        return train_loss.get_avg(), train_top1_acc.get_avg(), train_top5_acc.get_avg()

    def validate(self):
        return predict(
            self.model, self.device, self.val_loader, self.loss_function)


class CLIPDistillTrainer(Trainer):
    def __init__(
            self, model: nn.Module, train_loader: DataLoader, 
            val_loader: DataLoader, device: torch.device, logger: Logger,
            args: Namespace):
        super().__init__(
            model, train_loader, val_loader, device, logger, args)       
        
        self.augment = args.augmented_distillation 
        if self.augment:
            self.transform = SimCLRTransform()

    def train_epoch(self, epoch):
        self.model.student.train()
        self.model.teacher.eval()
        train_loss = AverageMeter()        
        for batch_idx, (data, _) in enumerate(self.train_loader):                                                            
            self.update_lr()
            data = data.to(self.device, non_blocking = True)
            if self.augment:
                data = self.transform(data, num_views=2)
                data = torch.cat(data, dim=0)                
            
            self.optimizer.zero_grad(set_to_none=True)            
            teacher_logits, student_logits = self.model(data)
            loss = self.compute_loss(teacher_logits, student_logits)            
            train_loss.update(loss.item())
            loss.backward()
            self.optimizer.step()                       
            
            if batch_idx % self.log_interval == 0:
                logged_loss = train_loss.get_avg()
                self.iters.append((epoch-1) + batch_idx / len(self.train_loader))
                self.logged_train_losses.append(logged_loss)
                log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                    epoch, batch_idx * len(data), int(len(self.train_loader.dataset)),
                    100. * batch_idx / len(self.train_loader), logged_loss)
                self.logger.log(log_str)
            if batch_idx % self.plot_interval == 0 and self.logger.save:
                train_loss_plot(
                    self.iters, self.logged_train_losses, self.plots_dir)
            self.itr += 1

        return train_loss.get_avg(), None, None

    def validate(self):
        self.model.eval()        
        loss = 0
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking = True)
                teacher_logits, student_logits = self.model(data)
                loss += self.compute_loss(teacher_logits, student_logits).item()                                

        loss = loss / len(self.val_loader)

        return loss, None, None

    def compute_loss(self, teacher_logits,student_logits):            
        targets = torch.arange(teacher_logits.shape[0], device = self.device)
        ce_teacher = F.cross_entropy(teacher_logits,targets)
        ce_student = F.cross_entropy(student_logits,targets)    
        
        return (ce_teacher + ce_student) / 2
        # targets = torch.arange(teacher_logits.shape[0], device = self.device)        
        # ce_student = F.cross_entropy(student_logits,targets)    
        
        return ce_student



class DistillationTrainer(Trainer):
    def __init__(
            self, model: nn.Module, teacher: nn.Module, train_loader: DataLoader, 
            val_loader: DataLoader, device: torch.device, logger: Logger,
            args: Namespace):
        super().__init__(
            model, train_loader, val_loader, device, logger, args)
        self.student = self.model
        self.teacher = teacher
        self.margin = args.margin
        self.margin_type = args.margin_type
        self.margin_value = args.margin_value
        self.loss_type = args.distillation_loss
        
        if self.loss_type == 'similarity-based':
            self.distiller = SimilarityDistiller(args.augmented_distillation, self.margin, self.margin_value,self.margin_type, args.sup_term,args.c)
        elif self.loss_type == 'similarity-weighted':
            self.distiller = WeightedDistiller(args.teacher_temp)
        else:
            self.distiller = KD(args.c)

    def train_epoch(self, epoch):
        self.student.train()      
        self.teacher.eval()
        train_loss = AverageMeter()        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.update_lr()
            data = data.to(self.device, non_blocking = True)
            target = target.to(self.device, non_blocking = True)     

            # n = target.shape[0]
            # ppair = torch.sum(target.repeat(n,1) == target.unsqueeze(1).repeat(1,n))
            # print('*' * 80)
            # print(ppair / n**2)


            self.optimizer.zero_grad(set_to_none=True)            
            loss = self.distiller.compute_loss(self.student, self.teacher, data, target)
            train_loss.update(loss.item())
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                logged_loss = train_loss.get_avg()
                self.iters.append((epoch-1) + batch_idx / len(self.train_loader))
                self.logged_train_losses.append(logged_loss)
                log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                    epoch, batch_idx * len(data), int(len(self.train_loader.dataset)),
                    100. * batch_idx / len(self.train_loader), logged_loss)
                self.logger.log(log_str)
            if batch_idx % self.plot_interval == 0 and self.logger.save:
                train_loss_plot(
                    self.iters, self.logged_train_losses, self.plots_dir)
            self.itr += 1
        
            # self.scheduler.step()            
        
        return train_loss.get_avg(), None, None

    def validate(self):
        self.model.eval()
        self.teacher.eval()
        loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device, non_blocking = True)
                target = target.to(self.device, non_blocking = True)
                loss += self.distiller.compute_loss(self.student, self.teacher, data, target).item()

        loss = loss / len(self.val_loader)

        return loss, None, None

def predict(model: nn.Module, device: torch.device, 
    loader: torch.utils.data.DataLoader, loss_function: nn.Module) -> tuple([float, float]):
    """Evaluate supervised model on data.

    Args:
        model: Model to be evaluated.
        device: Device to evaluate on.
        loader: Data to evaluate on.
        loss_function: Loss function being used.
        precision: precision to evaluate model with

    Returns:
        Model loss and accuracy on the evaluation dataset.
    
    """
    model.eval()     
    

    loss = 0
    acc1 = 0
    acc5 = 0
    with torch.no_grad():
        for data, target in loader:            
            data, target = data.to(device, non_blocking = True), target.to(device, non_blocking = True)
            output = model(data)         
            loss += loss_function(output, target).item()

            cur_acc1, cur_acc5 = compute_accuracy(output, target)
            acc1 += cur_acc1
            acc5 += cur_acc5
    
    loss, acc1, acc5 = loss / len(loader), acc1 / len(loader), acc5 / len(loader)
   
    return loss, acc1, acc5


def to_precision(object,precision):
    '''Move given object to given precision. Works with model + tensors'''
    if (precision == '32') or (precision == 'autocast'):
        pass    
    elif precision == '16':
        object = object.half()    
    elif precision == '8':                
        if isinstance(object,nn.Module):                        
            object = torch.quantization.quantize_dynamic(
                object, {nn.Conv2d,nn.Linear}, dtype=torch.qint8)              
        
    return object


def print_size_of_model(model, label=""):
    import os
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size
        
def compute_confusion(output,target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t().squeeze(0)        

        confusion = confusion_matrix(target.cpu().numpy(),pred.cpu().numpy(),normalize='true',labels = range(10) )
    return confusion

def compute_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.shape[0]

        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        top1_acc = correct[:1].view(-1).float().sum(0, keepdim=True) * 100.0 / batch_size
        top5_acc = correct[:5].reshape(-1).float().sum(0, keepdim=True) * 100.0 / batch_size

    return top1_acc.item(), top5_acc.item()

def get_embeddings(model: nn.Module, device: torch.device, 
    loader: torch.utils.data.DataLoader, emb_dim: int
    ) -> tuple([np.ndarray, np.ndarray]):
    """Get model's embeddings on data in numpy format.
    
    Args:
        model: The model computing the embeddings.
        device: The device on which the model and data are stored.
        loader: The input.
        emb_dim: The embedding dimension.

    Returns:
        The embeddings and labels for all instances in the data.
    
    """    
    model.to(device)
    model.eval()

    embeddings = np.zeros((0, emb_dim))
    labels = np.zeros((0))

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)            
            embs = model(data)
            embeddings = np.concatenate((embeddings, embs.cpu().numpy()))
            labels = np.concatenate((labels, target))

    return embeddings, labels

def get_labels(loader: torch.utils.data.DataLoader) -> np.ndarray:
    """Extract only the labels as a numpy array.

    Args:
        loader: The labelled data.

    Returns:
        The labels for the data.

    """
    labels = np.zeros((0))
    for _, target in loader:
        labels = np.concatenate((labels, target))

    return labels
    
def get_model_similarity(model: nn.Module, data: torch.Tensor, 
    augmented_data: torch.Tensor) -> torch.Tensor:
    """Get similarities between original and augmented data, as predicted by the model. 

    Args:
        model: The model.
        data: The input.
        augmented_data: The augmented input.

    Returns:
        The similarities output by the model.

    """
    #Get embeddings of original data
    data_embs = model(data)
    #Get embeddings of augmented data
    augmented_data_embs = model(augmented_data)

    data_embs = F.normalize(data_embs, p=2, dim=1)
    augmented_data_embs = F.normalize(augmented_data_embs, p=2, dim=1)

    return torch.sum(data_embs * augmented_data_embs, dim=1)

def train_loss_plot(iters: list([float]), train_vals: list([float]), 
    save_dir: str) -> None:
    plt.figure()
    plt.plot(iters, train_vals, 'b-')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.savefig(save_dir + '/train_loss_plot.png')
    plt.close()

def train_val_plots(train_vals: list([float]), val_vals: list([float]), 
    y_label: str, save_dir: str, change_epochs: list([int])) -> None:
    """Plotting loss or accuracy as a function of epoch number.

    Args:
        train_vals: y-axis training values (loss or accuracy).
        val_vals: y-axis validation values (loss or accuracy).
        y_label: y-axis label (loss or accuracy).
        save_dir: Directory where plots will be saved.
        change_epochs: Epochs where learning rate changes.

    """
    epochs = np.arange(1,len(train_vals)+1)
    plt.figure()
    plt.plot(epochs, train_vals, 'b-')
    plt.plot(epochs, val_vals, 'r-')
    if change_epochs is not None:
        for e in change_epochs:
            plt.axvline(e, linestyle='--', color='0.8')
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.savefig(save_dir + '/' + y_label + '_plots')
    plt.close()

class AverageMeter():
    """Computes and stores the average and current value
    
    Taken from the Torch examples repository:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg