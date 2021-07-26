"""Code for training loops and loss computation.

Supports supervised training, similarity-based embedding training,
and similarity-based distillation training.

Includes code for training summary and plots.

Based on code from PyTorch example
https://github.com/pytorch/examples/blob/master/mnist/main.py

"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from matplotlib import pyplot as plt
from data_augmentation import augment
from logger import Logger

def train_sup(model: nn.Module, train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader, device: torch.device, 
    loss_function: nn.Module = nn.CrossEntropyLoss, epochs: int = 200, 
    lr: float = 0.1, optimizer_choice: str = 'adam', 
    scheduler_choice: str = 'plateau', patience: int = 5, 
    early_stop: int = 10, log_interval: int = 10, logger: Logger = None, 
    rank: int = 0, num_devices: int = 1) -> tuple([list([float]), list([float]),
    list([float]), list([float])]):
    """Supervised training loop.

    Args:
        model: The model to be trained.
        train_loader: The training set.
        valid_loader: The validation set.
        device: Device to train on. Recommended to use CUDA if it is available.
        loss_function: Loss function to use.
        epochs: Maximum number of epochs.
        lr: Starting learning rate.
        optimizer_choice: Choice of optimizer (Adam or Momentum SGD).
        scheduler_choice: Choice of scheduler (Plateau or Cosine Annealing).
        patience: Scheduler patience (applies only to Plateau scheduler).
        early_stop: Early stopping patience.
        log_interval: Number of batches between logs.
        logger: Logger object which tracks live training information.
        rank: Rank of the current process (useful for multiprocessing).
        num_devices: Number of devices.

    Returns:
        List of training losses, list of training accuracies, list of 
        validation losses, list of validation accuracies (one entry
        per epoch).

    Raises:
        ValueError: Only Adam and SGD optimizers are supported.
        ValueError: Only Plateau and Cosine schedulers are supported.
    
    """
    save_path = logger.get_model_path()

    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('Only Adam and SGD optimizers are supported.')

    if scheduler_choice == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        plateau_factor = 0.1
    elif scheduler_choice == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        raise ValueError('Only Plateau and Cosine schedulers are supported.')

    #Keep track of losses and accuracies
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs_until_stop = early_stop

    #Keep track of changes in learning rate
    change_epochs = []

    for epoch in range(1, epochs + 1):
        train_sup_epoch(model, device, train_loader, loss_function, optimizer, epoch, 
            log_interval, logger, rank, num_devices)
        #train_loss, train_acc = predict(model, device, train_loader, loss_function, logger, "Train")
        val_loss, val_acc = predict(model, device, valid_loader, loss_function, logger, "Validation")

        #train_losses.append(train_loss)
        #train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        #Check if validation loss is worsening
        if val_loss > min(val_losses):
            epochs_until_stop -= 1
            if epochs_until_stop == 0: #Early stopping initiated
                break
        else:
            epochs_until_stop = early_stop

            #Save the current model (as it is the best one so far)
            if rank == 0:
                if logger.get_model_path() is not None:
                    logger.log("Saving model...")
                    if num_devices > 1:
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
                    logger.log("Model saved.\n")

        if scheduler_choice == 'plateau':
            scheduler.step(val_loss)
            if optimizer.param_groups[0]['lr'] == plateau_factor * lr:
                change_epochs.append(epoch)
                lr = plateau_factor * lr
                logger.log("Learning rate decreasing to {}\n".format(lr))
        else:
            scheduler.step()
            
    if rank == 0:
        train_report(train_losses, val_losses, train_accs, val_accs, change_epochs=change_epochs, logger=logger)

    return train_losses, train_accs, val_losses, val_accs

def train_distillation(student: nn.Module, teacher: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    valid_loader: torch.utils.data.DataLoader, device: torch.device, 
    loss_function: nn.Module = nn.MSELoss, epochs: int = 200, 
    lr: float = 0.1, optimizer_choice: str = 'adam', 
    scheduler_choice: str = 'plateau', patience: int = 5, 
    early_stop: int = 10, log_interval: int = 10, 
    logger: Logger = None, cosine: bool = False,
    rank: int = 0, num_devices: int = 1) -> tuple([list([float]), list([float])]):
    """Distillation training loop.

    Args:
        student: Student model to be trained.
        teacher: Teacher model to emulate.
        train_loader: The training set.
        valid_loader: The validation set.
        device: Device to train on. Recommended to use CUDA if it is available.
        loss_function: Loss function to minimize.
        epochs: Maximum number of epochs.
        lr: Starting learning rate.
        optimizer_choice: Choice of optimizer (Adam or Momentum SGD).
        scheduler_choice: Choice of scheduler (Plateau or Cosine Annealing).
        patience: Scheduler patience (applies only to Plateau scheduler).
        early_stop: Early stopping patience.
        log_interval: Number of batches between logs.
        logger: Logger object which tracks live training information.
        cosine: Set to true to match cosine similarities, otherwise match dot products.
        rank: Rank of the current process (useful for multiprocessing).
        num_devices: Number of devices.

    Returns:
        List of training losses, list of validation losses (one entry
        per epoch).

    Raises:
        ValueError: Only Adam and SGD optimizers are supported.
        ValueError: Only Plateau and Cosine schedulers are supported.
    
    """
    student = student.to(device)
    teacher = teacher.to(device)

    save_path = logger.get_model_path()

    if optimizer_choice == 'adam':
        optimizer = optim.Adam(student.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('Only Adam and SGD optimizers are supported.')

    if scheduler_choice == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        plateau_factor = 0.1
    elif scheduler_choice == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        raise ValueError('Only Plateau and Cosine schedulers are supported.')

    #Keep track of losses and accuracies
    train_losses = []
    val_losses = []
    epochs_until_stop = early_stop

    change_epochs = []

    for epoch in range(1, epochs + 1):
        train_distillation_epoch(student, teacher, device, train_loader, loss_function, 
            optimizer, epoch, log_interval, logger, cosine, rank, num_devices)
        #train_loss = compute_distillation_loss(student, teacher, device, train_loader, 
            #loss_function, cosine, logger, "Training")
        val_loss = compute_distillation_loss(student, teacher, device, valid_loader, 
            loss_function, cosine, logger, "Validation")

        #train_losses.append(train_loss)
        val_losses.append(val_loss)

        #Check if validation loss is worsening
        if val_loss > min(val_losses):
            epochs_until_stop -= 1
            if epochs_until_stop == 0: #Early stopping initiated
                break
        else:
            epochs_until_stop = early_stop

            #Save the current model (as it is the best one so far)
            if rank == 0:
                if save_path is not None:
                    logger.log("Saving model...")
                    if num_devices > 1:
                        torch.save(student.module.state_dict(), save_path)
                    else:
                        torch.save(student.state_dict(), save_path)
                    logger.log("Model saved.\n")

        scheduler.step(val_loss)

        if scheduler_choice == 'plateau':
            if optimizer.param_groups[0]['lr'] == plateau_factor * lr:
                change_epochs.append(epoch)
                lr = plateau_factor * lr
                logger.log("Learning rate decreasing to {}\n".format(lr))

    if rank == 0:
        train_report(train_losses, val_losses, change_epochs=change_epochs, logger=logger)

    return train_losses, val_losses

def train_similarity(model: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    valid_loader: torch.utils.data.DataLoader, device: torch.device, 
    augmentation: str = 'blur-sigma', alpha_max: float = 15.0, 
    beta: float = 0.2, loss_function: nn.Module = nn.MSELoss, 
    epochs: int = 200, lr: float = 0.1, optimizer_choice: str = 'adam', 
    scheduler_choice: str = 'plateau', patience: int = 5, 
    early_stop: int = 10, log_interval: int = 10, logger: Logger = None, 
    cosine: bool = False, temp: float = 0.01
    ) -> tuple([list([float]), list([float])]):
    """Similarity-based embedding training loop.

    Args:
        model: Model to be trained.
        train_loader: The training set.
        valid_loader: The validation set.
        device: Device to train on. Recommended to use CUDA if it is available.
        augmentation: Type of data augmentation to use.
        alpha_max: Maximum intensity of data augmentation.
        beta: Parameter of similarity probability function.
        loss_function: Loss function to minimize (MSE or KL divergence).
        epochs: Maximum number of epochs.
        lr: Starting learning rate.
        optimizer_choice: Choice of optimizer (Adam or Momentum SGD).
        scheduler_choice: Choice of scheduler (Plateau or Cosine Annealing).
        patience: Scheduler patience (applies only to Plateau scheduler).
        early_stop: Early stopping patience.
        log_interval: Number of batches between logs.
        logger: Logger object which tracks live training information.
        cosine: Set to true to match cosine similarities, otherwise match dot products.
        temp: Temperature hyperparameter in sigmoid.

    Returns:
        List of training losses, list of validation losses (one entry
        per epoch).

    Raises:
        ValueError: Only Adam and SGD optimizers are supported.
        ValueError: Only Plateau and Cosine schedulers are supported.
    
    """
    model = model.to(device)

    save_path = logger.get_model_path()

    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('Only Adam and SGD optimizers are supported.')

    if scheduler_choice == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        plateau_factor = 0.1
    elif scheduler_choice == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        raise ValueError('Only Plateau and Cosine schedulers are supported.')

    #Keep track of losses and accuracies
    train_losses = []
    val_losses = []
    epochs_until_stop = early_stop

    change_epochs = []

    for epoch in range(1, epochs + 1):
        train_similarity_epoch(model, device, train_loader, loss_function, 
            optimizer, epoch, log_interval, logger, augmentation, alpha_max, beta, cosine, temp)
        train_loss = compute_similarity_loss(model, device, train_loader, loss_function, 
            augmentation, alpha_max, beta, cosine, temp, logger, "Train")
        val_loss = compute_similarity_loss(model, device, valid_loader, loss_function, 
            augmentation, alpha_max, beta, cosine, temp, logger, "Validation")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #Check if validation loss is worsening
        if val_loss > min(val_losses):
            epochs_until_stop -= 1
            if epochs_until_stop == 0: #Early stopping initiated
                break
        else:
            epochs_until_stop = early_stop

            #Save the current model (as it is the best one so far)
            if save_path is not None:
                logger.log("Saving model...")
                torch.save(model.state_dict(), save_path)
                logger.log("Model saved.\n")

        scheduler.step(val_loss)

        if scheduler_choice == 'plateau':
            if optimizer.param_groups[0]['lr'] == plateau_factor * lr:
                change_epochs.append(epoch)
                lr = plateau_factor * lr
                logger.log("Learning rate decreasing to {}\n".format(lr))

    train_report(train_losses, val_losses, change_epochs=change_epochs, logger=logger)

    return train_losses, val_losses

def train_sup_epoch(model: nn.Module, device: torch.device, 
    train_loader: torch.utils.data.DataLoader, loss_function: nn.Module, 
    optimizer: optim.Optimizer, epoch: int, log_interval: int, 
    logger: Logger, rank: int, num_devices: int) -> None:
    """Train in supervised fashion for one epoch.

    Args:
        model: Model being trained.
        train_loader: The training set.
        device: Device to train on. Recommended to use CUDA if it is available.
        loss_function: Loss function to minimize.
        optimizer: Optimizer being used.
        epoch: The current epoch number.
        log_interval: Number of batches between logs.
        logger: Logger object which tracks live training information.
        rank: Rank of the current device.
        num_devices: Number of devices used for training.
    
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                epoch, batch_idx * len(data), int(len(train_loader.dataset) / num_devices),
                100. * batch_idx / len(train_loader), loss.item())
            logger.log(log_str)

def train_distillation_epoch(student: nn.Module, teacher: nn.Module,
    device: torch.device, train_loader: torch.utils.data.DataLoader, 
    loss_function: nn.Module, optimizer: optim.Optimizer, epoch: int, 
    log_interval: int, logger: Logger, cosine: bool,
    rank: int, num_devices: int) -> None:
    """Train the student for one epoch.

    Args:
        student: Student model to be trained.
        teacher: Teacher model to emulate.
        device: Device to train on. 
        train_loader: The training set.
        loss_function: Loss function to minimize.
        optimizer: Optimizer being used.
        epoch: The current epoch number.
        log_interval: Number of batches between logs.
        logger: Logger object which tracks live training information.
        cosine: Set to true to match cosine similarities, otherwise match dot products.
        rank: Rank of the current device.
        num_devices: Number of devices used for training.

    """
    student.train()
    teacher.eval()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        student_sims, teacher_sims = get_student_teacher_similarity(student, teacher, data, cosine)
        loss = loss_function(student_sims, teacher_sims)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                epoch, batch_idx * len(data), int(len(train_loader.dataset) / num_devices),
                100. * batch_idx / len(train_loader), loss.item())
            logger.log(log_str)

#Epoch of similarity training
def train_similarity_epoch(model: nn.Module, device: torch.device, 
    train_loader: torch.utils.data.DataLoader, loss_function: nn.Module, 
    optimizer: optim.Optimizer, epoch: int, log_interval: int, 
    logger: Logger, augmentation: str, alpha_max: float, beta: float, 
    cosine: bool, temp: float) -> None:
    """Train similarity-based embeddings for one epoch.

    Args:
        model: Model to be trained.
        device: Device to train on. 
        train_loader: The training set.
        loss_function: Loss function to minimize.
        optimizer: Optimizer being used.
        epoch: The current epoch number.
        log_interval: Number of batches between logs.
        logger: Logger object which tracks live training information.
        augmentation: Type of data augmentation being used.
        alpha_max: Largest possible augmentation intensity.
        beta: Parameter of similarity probability.
        cosine: Set to true to match cosine similarities, otherwise match dot products.
        temp: Temperature hyperparameter for sigmoid.

    """
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        #Get augmented data, target probabilities, and model probabilities
        with torch.no_grad():
            augmented_data, target = augment(data, augmentation, device, alpha_max, beta)
        model_sims = get_model_similarity(model, data, augmented_data, cosine)

        if isinstance(loss_function, nn.KLDivLoss):
            output = get_model_probability(model_sims, temp)
            eps = 1e-7 #Constant to prevent log(0)
            output_comp = 1 - output
            output = torch.stack((output, output_comp), dim=1)
            #KL Divergence expects output to be log probability distribution
            output = (output + eps).log()
            target_comp = 1 - target
            target = torch.stack((target, target_comp), dim=1)
        else:
            output = model_sims

        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            logger.log(log_str)

def predict(model: nn.Module, device: torch.device, 
    loader: torch.utils.data.DataLoader, loss_function: nn.Module, 
    logger: Logger, subset: str) -> tuple([float, float]):
    """Evaluate supervised model on data.

    Args:
        model: Model to be evaluated.
        device: Device to evaluate on.
        loader: Data to evaluate on.
        loss_function: Loss function being used.
        logger: Logger object which tracks model performance.

    Returns:
        Model loss and accuracy on the evaluation dataset.
    
    """
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(loss_function(output, target).item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    loss = np.mean(losses)

    acc = 100.00 * correct/ len(loader.dataset)

    log_str = '\n{} set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(subset,
        loss, correct, len(loader.dataset), acc)
    logger.log(log_str)

    return loss, acc

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
    loss_function: nn.Module, cosine: bool, logger: Logger, subset: str
    ) -> float:
    """Compute the distillation loss.

    Args:
        student: The student model.
        teacher: The teacher model.
        device: The device that the models and data are stored on.
        loader: The data the loss is computed on.
        loss_function: The loss function.
        cosine: Whether or not cosine similarity is being used.
        logger: Logs the loss.
        subset: Indicates which set the data belongs to (training or validation).

    Returns:
        The distillation loss.

    """
    student.eval()
    teacher.eval()
    losses = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            student_sims, teacher_sims = get_student_teacher_similarity(student, teacher, data, cosine)
            losses.append(loss_function(student_sims, teacher_sims).item())

    loss = np.mean(losses)

    log_str = '\n{} set: Average loss: {:.6f}\n'.format(subset, loss)
    logger.log(log_str)

    return loss

def compute_similarity_loss(model: nn.Module, device: torch.device, 
    loader: torch.utils.data.DataLoader, loss_function: nn.Module, 
    augmentation: str, alpha_max: float, beta: float, cosine: bool, 
    temp: float, logger: Logger, subset: str) -> float:
    """Compute the similarity-based embedding loss.

    Args:
        model: The model being evaluated.
        device: The device that the model and data are stored on.
        loader: The data.
        loss_function: The loss function.
        Augmentation: Type of data augmentation being used.
        alpha_max: Maximum intensity of data augmentation.
        beta: Parameter of similarity probability
        cosine: Whether or not cosine similarity is being used.
        temp: Temperature hyperparameter for sigmoid.
        logger: Logs the loss.
        subset: Indicates which set the data belongs to (training or validation).

    Returns:
        The similarity loss.

    """
    model.eval()
    losses = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            augmented_data, target = augment(data, augmentation, device, alpha_max, beta)
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
                
            losses.append(loss_function(output, target).item())

    loss = np.mean(losses)

    log_str = '\n{} set: Average loss: {:.6f}\n'.format(subset, loss)
    logger.log(log_str)

    return loss
            
def train_report(train_losses: list([float]), val_losses: list([float]), 
    train_accs: list([float]) = None, val_accs: list([float]) = None, 
    change_epochs: list([int]) = None, logger: Logger = None) -> None:
    """Produce a summary of training.

    Args:
        train_losses: Training set loss after each epoch.
        val_losses: Validation set loss after each epoch.
        train_accs: Training accuracies after each epoch.
        val_accs: Validation accuracies after each epoch.
        change_epochs: List of epochs where learning rate changes.
        logger: The results logger.

    """
    best_epoch = np.argmin(val_losses)
    logger.log("Training complete.\n")
    logger.log_results("Best Epoch: {}".format(best_epoch + 1))
    logger.log_results("Training Loss: {:.6f}".format(train_losses[best_epoch]))
    if train_accs is not None:
        logger.log_results("Training Accuracy: {:.2f}".format(train_accs[best_epoch]))
    logger.log_results("Validation Loss: {:.6f}".format(val_losses[best_epoch]))
    if val_accs is not None:
        logger.log_results("Validation Accuracy: {:.2f}".format(val_accs[best_epoch]))

    #Save loss and accuracy plots
    save_dir = logger.get_plots_dir()
    if save_dir is not None:
        train_plots(train_losses, val_losses, "Loss", save_dir, change_epochs)
        if train_accs is not None and val_accs is not None:
            train_plots(train_accs, val_accs, "Accuracy", save_dir, change_epochs)

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
