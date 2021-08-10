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
from matplotlib import pyplot as plt
from data_augmentation import make_augmentation, Augmentation
from logger import Logger

class Trainer():

    def __init__(
            self, model: nn.Module, train_loader: DataLoader, 
            val_loader: DataLoader, device: torch.device, 
            logger: Logger, loss_function: str, epochs: int = 250, 
            lr: float = 0.1, optim_str: str = 'adam', 
            sched_str: str = 'plateau', patience: int = 5, 
            early_stop: int = 10, log_interval: int = 10,
            rank: int = 0, num_devices: int = 1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        if loss_function == 'cross-entropy':
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function == 'mse':
            self.loss_function = nn.MSELoss()
        elif loss_function == 'kl':
            self.loss_function = nn.KLDivLoss()
        else:
            raise ValueError(
                'Only cross entropy, MSE, and KL divergence loss are supported.')
        self.epochs = epochs
        self.lr = lr

        if optim_str == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optim_str == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
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
        self.logger = logger
        self.save_path = logger.get_model_path()
        self.rank = rank
        self.num_devices = num_devices

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train(self):
        epochs_until_stop = self.early_stop
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            log_str = '\nTraining set: Average loss: {:.6f}\n'.format(
            train_loss)
            if train_acc is not None:
                log_str += 'Training set: Average accuracy: {:.2f}%\n'.format(train_acc)
            log_str += 'Validation set: Average loss: {:.6f}\n'.format(val_loss)
            if val_acc is not None:
                log_str += 'Validation set: Accuracy: {:.2f}\n'.format(val_acc)
            
            self.logger.log(log_str)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            #Check if validation loss is worsening
            if val_loss > min(self.val_losses):
                epochs_until_stop -= 1
                if epochs_until_stop == 0: #Early stopping initiated
                    break
            else:
                epochs_until_stop = self.early_stop

                #Save the current model (as it is the best one so far)
                if self.rank == 0:
                    if self.save_path is not None:
                        self.logger.log("Saving model...")
                        if self.num_devices > 1:
                            torch.save(self.model.module.state_dict(), self.save_path)
                        else:
                            torch.save(self.model.state_dict(), self.save_path)
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
            self.logger.log_results("Training Accuracy: {:.2f}".format(
                self.train_accs[best_epoch]))
        self.logger.log_results("Validation Loss: {:.6f}".format(
            self.val_losses[best_epoch]))
        if self.val_accs[best_epoch] is not None:
            self.logger.log_results("Validation Accuracy: {:.2f}".format(
                self.val_accs[best_epoch]))

        #Save loss and accuracy plots
        save_dir = self.logger.get_plots_dir()
        if save_dir is not None:
            train_plots(
                self.train_losses, self.val_losses, "Loss", save_dir, self.change_epochs)
            if self.train_accs is not None and self.val_accs is not None:
                train_plots(
                    self.train_accs, self.val_accs, "Accuracy", save_dir, self.change_epochs)

class SupervisedTrainer(Trainer):
    def __init__(
            self, model: nn.Module, train_loader: DataLoader, 
            val_loader: DataLoader, device: torch.device, 
            logger: Logger, loss_function: str, epochs: int = 250,
            lr: float = 0.1, optim_str: str = 'adam', 
            sched_str: str = 'plateau', patience: int = 5, 
            early_stop: int = 10, log_interval: int = 10,
            rank: int = 0, num_devices: int = 1):
        super().__init__(
            model, train_loader, val_loader, device, logger, loss_function,
            epochs, lr, optim_str, sched_str, patience, early_stop,
            log_interval, rank, num_devices)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            acc = compute_accuracy(output, target)
            train_loss.update(loss.item())
            train_acc.update(acc)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                    epoch, batch_idx * len(data), int(len(self.train_loader.dataset) / self.num_devices),
                    100. * batch_idx / len(self.train_loader), loss.item())
                self.logger.log(log_str)

        return train_loss.get_avg(), train_acc.get_avg()

    def validate(self):
        return predict(
            self.model, self.device, self.val_loader, self.loss_function)

class DistillationTrainer(Trainer):
    def __init__(
            self, model: nn.Module, teacher: nn.Module,
            train_loader: DataLoader, val_loader: DataLoader, 
            device: torch.device, logger: Logger, 
            loss_function: str, epochs: int = 250,
            lr: float = 0.1, optim_str: str = 'adam', 
            sched_str: str = 'plateau', patience: int = 5, 
            early_stop: int = 10, log_interval: int = 10,
            rank: int = 0, num_devices: int = 1,
            cosine: bool = True):
        super().__init__(
            model, train_loader, val_loader, device, logger, loss_function,
            epochs, lr, optim_str, sched_str, patience, early_stop,
            log_interval, rank, num_devices)
        self.teacher = teacher
        self.cosine = cosine

    def train_epoch(self, epoch):
        self.model.train()
        self.teacher.eval()
        train_loss = AverageMeter()
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            student_sims, teacher_sims = get_student_teacher_similarity(
                self.model, self.teacher, data, self.cosine)
            loss = self.loss_function(student_sims, teacher_sims)
            train_loss.update(loss.item())
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                    epoch, batch_idx * len(data), int(len(self.train_loader.dataset) / self.num_devices),
                    100. * batch_idx / len(self.train_loader), loss.item())
                self.logger.log(log_str)
        
        return train_loss.get_avg(), None

    def validate(self):
        return compute_distillation_loss(
            self.model, self.teacher, self.device, self.val_loader, 
            self.loss_function, self.cosine), None

class SimilarityTrainer(Trainer):
    def __init__(
            self, model: nn.Module, aug: str, alpha_max: float,
            kernel_size: int, beta: float, train_loader: DataLoader, 
            val_loader: DataLoader, device: torch.device, 
            logger: Logger, loss_function: str, epochs: int = 250,
            lr: float = 0.1, optim_str: str = 'adam', 
            sched_str: str = 'plateau', patience: int = 5, 
            early_stop: int = 10, log_interval: int = 10,
            rank: int = 0, num_devices: int = 1,
            temp: float = 0.01):
        super().__init__(
            model, train_loader, val_loader, device, logger, loss_function,
            epochs, lr, optim_str, sched_str, patience, early_stop,
            log_interval, rank, num_devices)
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
                log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item())
                self.logger.log(log_str)

        return train_loss.get_avg(), None

    def validate(self):
        return compute_similarity_loss(
            self.model, self.device, self.val_loader, self.loss_function,
            self.aug, self.alpha_max, self.beta, self.cosine, self.temp), None

def get_trainer(mode: str, model: nn.Module, train_loader: DataLoader, 
        val_loader: DataLoader, device: torch.device, 
        logger: Logger, loss_function: str, epochs: int = 250, 
        lr: float = 0.1, optim_str: str = 'adam', 
        sched_str: str = 'plateau', patience: int = 5, 
        early_stop: int = 10, log_interval: int = 10,
        rank: int = 0, num_devices: int = 1, 
        teacher_model: nn.Module = None, cosine: bool = True,
        aug: str = None, alpha_max: float = None, 
        kernel_size: int = None, beta: float = None, temp: float = None):
    if mode == 'teacher' or mode == 'linear_classifier' or mode == 'random':
        return SupervisedTrainer(
            model, train_loader, val_loader, device, logger, loss_function,
            epochs, lr, optim_str, sched_str, patience, early_stop,
            log_interval, rank, num_devices)
    elif mode == 'distillation':
        return DistillationTrainer(
            model, teacher_model, train_loader, val_loader, device,
            logger, loss_function, epochs, lr, optim_str, sched_str,
            patience, early_stop, log_interval, rank, num_devices,
            cosine)
    else:
        return SimilarityTrainer(
            model, aug, alpha_max, kernel_size, beta, train_loader, val_loader,
            device, logger, loss_function, epochs, lr, optim_str, sched_str,
            patience, early_stop, log_interval, rank, num_devices, temp)

def predict(model: nn.Module, device: torch.device, 
    loader: torch.utils.data.DataLoader, loss_function: nn.Module) -> tuple([float, float]):
    """Evaluate supervised model on data.

    Args:
        model: Model to be evaluated.
        device: Device to evaluate on.
        loader: Data to evaluate on.
        loss_function: Loss function being used.

    Returns:
        Model loss and accuracy on the evaluation dataset.
    
    """
    model.eval()
    loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += loss_function(output, target).item()
            acc += compute_accuracy(output, target)
    
    loss, acc = loss / len(loader), acc / len(loader)

    return loss, acc

def compute_accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    acc = 100 * correct / output.shape[0]

    return acc

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

def compute_distillation_loss(student: nn.Module, teacher: nn.Module,
    device: torch.device, loader: torch.utils.data.DataLoader, 
    loss_function: nn.Module, cosine: bool) -> float:
    """Compute the distillation loss.

    Args:
        student: The student model.
        teacher: The teacher model.
        device: The device that the models and data are stored on.
        loader: The data the loss is computed on.
        loss_function: The loss function.
        cosine: Whether or not cosine similarity is being used.

    Returns:
        The distillation loss.

    """
    student.eval()
    teacher.eval()
    loss = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            student_sims, teacher_sims = get_student_teacher_similarity(student, teacher, data, cosine)
            loss += loss_function(student_sims, teacher_sims).item()

    loss = loss / len(loader)

    return loss

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

def train_plots(train_vals: list([float]), val_vals: list([float]), 
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