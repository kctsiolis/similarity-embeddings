"""Code for training loops and loss computation.

Supports supervised training, similarity-based embedding training,
and similarity-based distillation training.

Includes code for training summary and plots.

Loosely based on code from PyTorch example
https://github.com/pytorch/examples/blob/master/mnist/main.py

"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from data_augmentation import make_augmentation, Augmentation
from logger import Logger

class Trainer():

    def __init__(
            self, model: nn.Module, train_loader: DataLoader, 
            val_loader: DataLoader, validate: bool, 
            device: torch.device, logger: Logger, epochs: int = 250, 
            lr: float = 0.1, optim_str: str = 'adam', 
            sched_str: str = 'plateau', patience: int = 5, 
            early_stop: int = 10, log_interval: int = 10,
            plot_interval: int = None,
            rank: int = 0, num_devices: int = 1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_set = 'Validation' if validate else 'Test'
        self.device = device
        self.epochs = epochs
        self.lr = lr

        if optim_str == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optim_str == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError('Only Adam and SGD optimizers are supported.')

        self.sched_str = sched_str
        if sched_str == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=patience)
            self.plateau_factor = 0.1
            self.change_epochs = []
        elif sched_str == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200)
            self.change_epochs = None

        self.early_stop = early_stop
        self.log_interval = log_interval
        if plot_interval is None:
            self.plot_interval = len(self.train_loader) // 10
        else:
            self.plot_interval = plot_interval
        self.logger = logger
        self.model_path = logger.get_model_path()
        self.plots_dir = logger.get_plots_dir()
        self.rank = rank
        self.num_devices = num_devices

        self.iters = [] 
        self.logged_train_losses = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_accs5 = []
        self.val_accs5 = []

    def train(self):
        epochs_until_stop = self.early_stop
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc, train_acc5 = self.train_epoch(epoch)
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

            #Check if validation loss is worsening
            if val_loss > min(self.val_losses):
                epochs_until_stop -= 1
                if epochs_until_stop == 0: #Early stopping initiated
                    break
            else:
                epochs_until_stop = self.early_stop

                #Save the current model (as it is the best one so far)
                if self.rank == 0:
                    if self.model_path is not None:
                        self.logger.log("Saving model...")
                        if self.num_devices > 1:
                            torch.save(self.model.module.state_dict(), self.model_path)
                        else:
                            torch.save(self.model.state_dict(), self.model_path)
                        self.logger.log("Model saved.\n")

            if self.sched_str == 'plateau':
                self.scheduler.step(val_loss)
                if self.optimizer.param_groups[0]['lr'] == self.plateau_factor * self.lr:
                    self.change_epochs.append(epoch)
                    lr = self.plateau_factor * self.lr
                    self.logger.log("Learning rate decreasing to {}\n".format(lr))
            else:
                self.scheduler.step()

        if self.rank == 0:
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
        train_val_plots(
            self.train_losses, self.val_losses, "Loss", self.plots_dir, self.change_epochs)
        if self.train_accs is not None and self.val_accs is not None:
            train_val_plots(
                self.train_accs, self.val_accs, "Accuracy", self.plots_dir, self.change_epochs)

class SupervisedTrainer(Trainer):
    def __init__(
            self, model: nn.Module, train_loader: DataLoader, 
            val_loader: DataLoader, validate: bool, 
            device: torch.device, logger: Logger, epochs: int = 250,
            lr: float = 0.1, optim_str: str = 'adam', 
            sched_str: str = 'plateau', patience: int = 5, 
            early_stop: int = 10, log_interval: int = 10,
            plot_interval: int = None, rank: int = 0, 
            num_devices: int = 1):
        super().__init__(
            model, train_loader, val_loader, validate, device, logger,
            epochs, lr, optim_str, sched_str, patience, early_stop,
            log_interval, plot_interval, rank, num_devices)
        self.loss_function = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = AverageMeter()
        train_top1_acc = AverageMeter()
        train_top5_acc = AverageMeter()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
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
                    epoch, batch_idx * len(data), int(len(self.train_loader.dataset) / self.num_devices),
                    100. * batch_idx / len(self.train_loader), logged_loss)
                self.logger.log(log_str)
            if batch_idx % self.plot_interval == 0:
                train_loss_plot(
                    self.iters, self.logged_train_losses, self.plots_dir)

        return train_loss.get_avg(), train_top1_acc.get_avg(), train_top5_acc.get_avg()

    def validate(self):
        return predict(
            self.model, self.device, self.val_loader, self.loss_function)

class DistillationTrainer(Trainer):
    def __init__(
            self, model: nn.Module, teacher: nn.Module,
            train_loader: DataLoader, val_loader: DataLoader, 
            validate: bool, device: torch.device, logger: Logger, 
            epochs: int = 250, lr: float = 0.1, 
            optim_str: str = 'adam', sched_str: str = 'plateau', 
            patience: int = 5, early_stop: int = 10, 
            log_interval: int = 10, plot_interval: int = None,
            rank: int = 0, num_devices: int = 1, cosine: bool = True, 
            distillation_type: str = 'similarity-based',
            c: float = 0.5):
        super().__init__(
            model, train_loader, val_loader, validate, device, logger,
            epochs, lr, optim_str, sched_str, patience, early_stop,
            log_interval, plot_interval, rank, num_devices)
        self.teacher = teacher
        self.cosine = cosine
        self.distillation_type = distillation_type
        if self.distillation_type == 'similarity-based':
            self.loss_function = nn.MSELoss()
        else:
            self.loss_function = nn.KLDivLoss(reduction='batchmean')
            self.sup_loss_function = nn.CrossEntropyLoss()
            if c < 0 or c > 1:
                raise ValueError('c must be in [0,1].')
            self.c = c

    def train_epoch(self, epoch):
        self.model.train()
        self.teacher.eval()
        train_loss = AverageMeter()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.compute_loss(data, target)
            train_loss.update(loss.item())
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                logged_loss = train_loss.get_avg()
                self.iters.append((epoch-1) + batch_idx / len(self.train_loader))
                self.logged_train_losses.append(logged_loss)
                log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                    epoch, batch_idx * len(data), int(len(self.train_loader.dataset) / self.num_devices),
                    100. * batch_idx / len(self.train_loader), logged_loss)
                self.logger.log(log_str)
            if batch_idx % self.plot_interval == 0:
                train_loss_plot(
                    self.iters, self.logged_train_losses, self.plots_dir)
        
        return train_loss.get_avg(), None, None

    def validate(self):
        self.model.eval()
        self.teacher.eval()
        loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                loss += self.compute_loss(data, target).item()

        loss = loss / len(self.val_loader)

        return loss, None, None

    def compute_loss(self, data, target):
        if self.distillation_type == 'similarity-based':
            student_sims, teacher_sims = get_student_teacher_similarity(
                self.model, self.teacher, data, self.cosine)
            loss = self.loss_function(student_sims, teacher_sims)
        else:
            output = self.model(data)
            model_probs = nn.LogSoftmax(dim=1)(output)
            teacher_probs = nn.Softmax(dim=1)(self.teacher(data))
            loss = self.c * self.loss_function(model_probs, teacher_probs) + \
                (1-self.c) * self.sup_loss_function(output, target)

        return loss

class SimilarityTrainer(Trainer):
    def __init__(
            self, model: nn.Module, aug: str, alpha_max: float,
            kernel_size: int, beta: float, train_loader: DataLoader, 
            val_loader: DataLoader, validate: bool, device: torch.device, 
            logger: Logger, epochs: int = 250,
            lr: float = 0.1, optim_str: str = 'adam', 
            sched_str: str = 'plateau', patience: int = 5, 
            early_stop: int = 10, log_interval: int = 10,
            plot_interval: int = None, rank: int = 0, num_devices: int = 1,
            temp: float = 0.01):
        super().__init__(
            model, train_loader, val_loader, validate, device, logger,
            epochs, lr, optim_str, sched_str, patience, early_stop,
            log_interval, plot_interval, rank, num_devices)
        self.augmentation = make_augmentation(
            aug, alpha_max=alpha_max, kernel_size=kernel_size, device=device)
        self.beta = beta
        self.temp = temp

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = AverageMeter()
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            #Get augmented data, target probabilities, and model probabilities
            with torch.no_grad():
                augmented_data, alpha = self.augmentation.augment(data)
                target = get_sim_prob(alpha, self.beta)
            model_sims = get_model_similarity(self.model, data, augmented_data, self.cosine)

            if isinstance(self.loss_function, nn.KLDivLoss):
                output = get_model_probability(model_sims, self.temp)
                eps = 1e-7 #Constant to prevent log(0)
                output_comp = 1 - output
                output = torch.stack((output, output_comp), dim=1)
                #KL Divergence expects output to be log probability distribution
                output = (output + eps).log()
                target_comp = 1 - target
                target = torch.stack((target, target_comp), dim=1)
            else:
                output = model_sims

            loss = self.loss_function(output, target)
            train_loss.update(loss.item())
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                logged_loss = train_loss.get_avg()
                self.iters.append((epoch-1) + batch_idx / len(self.train_loader))
                self.logged_train_losses.append(logged_loss)
                log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                    epoch, batch_idx * len(data), int(len(self.train_loader.dataset) / self.num_devices),
                    100. * batch_idx / len(self.train_loader), logged_loss)
                self.logger.log(log_str)
            if batch_idx % self.plot_interval == 0:
                train_loss_plot(
                    self.iters, self.logged_train_losses, self.plots_dir)

        return train_loss.get_avg(), None, None

    def validate(self):
        return compute_similarity_loss(
            self.model, self.device, self.val_loader, self.loss_function,
            self.aug, self.alpha_max, self.beta, self.cosine, self.temp), None, None

def get_trainer(mode: str, model: nn.Module, train_loader: DataLoader, 
        val_loader: DataLoader, validate: bool, device: torch.device, 
        logger: Logger, epochs: int = 250, 
        lr: float = 0.1, optim_str: str = 'adam', 
        sched_str: str = 'plateau', patience: int = 5, 
        early_stop: int = 10, log_interval: int = 10,
        plot_interval: int = None, rank: int = 0, num_devices: int = 1, 
        teacher_model: nn.Module = None, cosine: bool = True,
        distillation_type: str = 'similarity-based', c: float = 0.5,
        aug: str = None, alpha_max: float = None, 
        kernel_size: int = None, beta: float = None, temp: float = None):
    if mode == 'teacher' or mode == 'linear_classifier' or mode == 'random':
        return SupervisedTrainer(
            model, train_loader, val_loader, validate, device, logger,
            epochs, lr, optim_str, sched_str, patience, early_stop,
            log_interval, plot_interval, rank, num_devices)
    elif mode == 'distillation':
        return DistillationTrainer(
            model, teacher_model, train_loader, val_loader, validate, device,
            logger, epochs, lr, optim_str, sched_str,
            patience, early_stop, log_interval, plot_interval, rank, num_devices,
            cosine, distillation_type, c)
    else:
        return SimilarityTrainer(
            model, aug, alpha_max, kernel_size, beta, train_loader, val_loader,
            validate, device, logger, epochs, lr, optim_str, sched_str,
            patience, early_stop, log_interval, plot_interval, rank, num_devices, temp)

def predict(model: nn.Module, device: torch.device, 
    loader: torch.utils.data.DataLoader, loss_function: nn.Module, precision:str = '32',calculate_confusion:bool = False) -> tuple([float, float]):
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
    model = to_precision(model,precision)   
    

    loss = 0
    acc1 = 0
    acc5 = 0
    confusion = []
    with torch.no_grad():
        for data, target in loader:
            if precision == 'autocast':
                with torch.cuda.amp.autocast():
                    data, target = data.to(device),target.to(device)
                    output = model(data)            
                    loss += loss_function(output, target).item()
            else:
                data, target = to_precision(data.to(device),precision),target.to(device)
                output = model(data)            
                loss += loss_function(output, target).item()


            if calculate_confusion:
                confusion.append(compute_confusion(output,target))                
            
            cur_acc1, cur_acc5 = compute_accuracy(output, target)
            acc1 += cur_acc1
            acc5 += cur_acc5
    
    loss, acc1, acc5 = loss / len(loader), acc1 / len(loader), acc5 / len(loader)
    
    if calculate_confusion:                
        confusion = np.mean(np.stack(confusion),axis = 0)        
        return loss, acc1, acc5, confusion.round(2)
    else:
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

def get_student_teacher_similarity(student: nn.Module, 
    teacher: nn.Module, data: torch.Tensor, cosine: bool,
    ) -> tuple([torch.Tensor, torch.Tensor]):
    """Get similarities between instances, as given by student and teacher

    Args:
        student: The student model.
        teacher: The teacher model.
        data: The input.
        cosine: Set to true to compute cosine similarity, otherwise dot product.

    Returns:
        The similarities output by the student and the teacher.

    """
    student_embs = student(data)
    if cosine:
        student_embs = F.normalize(student_embs, p=2, dim=1)
    student_sims = torch.matmul(student_embs, student_embs.transpose(0,1))

    with torch.no_grad():
        teacher_embs = teacher(data)
        if cosine:
            teacher_embs = F.normalize(teacher_embs, p=2, dim=1)
        teacher_sims = torch.matmul(teacher_embs, teacher_embs.transpose(0,1))

    return student_sims, teacher_sims
    
def get_model_similarity(model: nn.Module, data: torch.Tensor, 
    augmented_data: torch.Tensor, cosine: bool) -> torch.Tensor:
    """Get similarities between original and augmented data, as predicted by the model. 

    Args:
        model: The model.
        data: The input.
        augmented_data: The augmented input.
        cosine: Set to true to compute cosine similarity, otherwise dot product.

    Returns:
        The similarities output by the model.

    """
    #Get embeddings of original data
    data_embs = model(data)
    #Get embeddings of augmented data
    augmented_data_embs = model(augmented_data)

    if cosine: #Using cosine similarity
        data_embs = F.normalize(data_embs, p=2, dim=1)
        augmented_data_embs = F.normalize(augmented_data_embs, p=2, dim=1)

    return torch.sum(torch.mul(data_embs, augmented_data_embs), dim=1)

def get_model_probability(sims: torch.Tensor, temp: float = 1.0
    ) -> torch.Tensor:
    """Get model's probability of similarity via sigmoid.

    Args:
        sims: Model similarities.
        temp: Temperature hyperparameter for sigmoid.

    Returns:
        Model's probability of similarity.

    """
    return 1 / (1 + torch.exp(-temp * sims))

def compute_similarity_loss(model: nn.Module, device: torch.device, 
    loader: torch.utils.data.DataLoader, loss_function: nn.Module, 
    augmentation: Augmentation, beta: float, cosine: bool, temp: float) -> float:
    """Compute the similarity-based embedding loss.

    Args:
        model: The model being evaluated.
        device: The device that the model and data are stored on.
        loader: The data.
        loss_function: The loss function.
        augmentation: Type of data augmentation being used.
        beta: Similarity probability parameter
        cosine: Whether or not cosine similarity is being used.
        temp: Temperature hyperparameter for sigmoid.

    Returns:
        The similarity loss.

    """
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            augmented_data, alpha = augmentation.augment(data)
            target = get_sim_prob(alpha, beta)
            model_sims = get_model_similarity(model, data, augmented_data, cosine)
            if isinstance(loss_function, nn.KLDivLoss):
                #KL Divergence expects output to be log probability distribution
                eps = 1e-7
                output = get_model_probability(model_sims, temp)
                output_comp = 1 - output
                output = torch.stack((output, output_comp), dim=1)
                output = (output + eps).log()
                target_comp = 1 - target
                target = torch.stack((target, target_comp), dim=1)
            else:
                output = model_sims
                
            loss += loss_function(output, target).item()

    loss = loss / len(loader)

    return loss

def get_sim_prob(alpha, beta):
    return torch.exp(-beta * alpha)

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