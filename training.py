#Useful functions for training neural nets
#Based on code from PyTorch example
#https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from data_augmentation import augment
from logger import Logger

#Supervised training loop
def train_sup(model, train_loader, valid_loader, device='cpu', train_batch_size=64, 
    valid_batch_size=1000, loss_function=nn.CrossEntropyLoss, epochs=20, lr=0.1,
    optimizer_choice='adam', patience=5, early_stop=5, log_interval=10, logger=None):

    device = torch.device(device)
    print("Running on device {}".format(device))
    model = model.to(device)

    save_path = logger.get_model_path()

    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('Only Adam and SGD optimizers are supported.')

    scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    plateau_factor = 0.1

    #Keep track of losses and accuracies
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs_until_stop = early_stop

    #Keep track of changes in learning rate
    change_epochs = []

    for epoch in range(1, epochs + 1):
        train_sup_epoch(model, device, train_loader, loss_function, optimizer, epoch, log_interval, logger)
        train_loss, train_acc = predict(model, device, train_loader, loss_function, logger, "Train")
        val_loss, val_acc = predict(model, device, valid_loader, loss_function, logger, "Validation")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
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
            if save_path is not None:
                logger.log("Saving model...")
                torch.save(model.state_dict(), save_path)
                logger.log("Model saved.\n")

        scheduler.step(val_loss)

        if optimizer.param_groups[0]['lr'] == plateau_factor * lr:
            change_epochs.append(epoch)
            lr = plateau_factor * lr
            logger.log("Learning rate decreasing to {}\n".format(lr))

    train_report(train_losses, val_losses, train_accs, val_accs, change_epochs=change_epochs, logger=logger)

    return train_losses, train_accs, val_losses, val_accs

def train_distillation(student, teacher, train_loader, valid_loader, device='cpu', 
    train_batch_size=64, valid_batch_size=1000, loss_function=nn.MSELoss, epochs=20, 
    lr=0.1, optimizer_choice='adam', patience=5, early_stop=5, log_interval=10, 
    logger=None, cosine=False):

    device = torch.device(device)
    print("Running on device {}".format(device))
    student = student.to(device)
    teacher = teacher.to(device)

    save_path = logger.get_model_path()

    if optimizer_choice == 'adam':
        optimizer = optim.Adam(student.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('Only Adam and SGD optimizers are supported.')

    scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    plateau_factor = 0.1

    #Keep track of losses and accuracies
    train_losses = []
    val_losses = []
    epochs_until_stop = early_stop

    change_epochs = []

    for epoch in range(1, epochs + 1):
        train_distillation_epoch(student, teacher, device, train_loader, loss_function, 
            optimizer, epoch, log_interval, logger, cosine)
        train_loss = compute_distillation_loss(student, teacher, device, train_loader, 
            loss_function, cosine, logger, "Training")
        val_loss = compute_distillation_loss(student, teacher, device, valid_loader, 
            loss_function, cosine, logger, "Validation")

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
                torch.save(student.state_dict(), save_path)
                logger.log("Model saved.\n")

        scheduler.step(val_loss)

        if optimizer.param_groups[0]['lr'] == plateau_factor * lr:
            change_epochs.append(epoch)
            lr = plateau_factor * lr
            logger.log("Learning rate decreasing to {}\n".format(lr))

    train_report(train_losses, val_losses, change_epochs=change_epochs, logger=logger)

    return train_losses, val_losses

#Supervised training loop
def train_similarity(model, train_loader, valid_loader, device='cpu', augmentation='blur-sigma',
    alpha_max=15, beta=0.2, train_batch_size=64, valid_batch_size=1000, loss_function=nn.MSELoss, epochs=50, 
    lr=0.1, optimizer_choice='adam', patience=5, early_stop=5, log_interval=10, logger=None, 
    cosine=False, temp=1):
    
    device = torch.device(device)
    print("Running on device {}".format(device))
    model = model.to(device)

    save_path = logger.get_model_path()

    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('Only Adam and SGD optimizers are supported.')

    scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    plateau_factor = 0.1

    #Keep track of losses and accuracies
    train_losses = []
    val_losses = []
    epochs_until_stop = early_stop

    change_epochs = []

    for epoch in range(1, epochs + 1):
        train_similarity_epoch(model, device, train_loader, train_batch_size, loss_function, 
            optimizer, epoch, log_interval, logger, augmentation, alpha_max, beta, cosine, temp)
        train_loss = compute_similarity_loss(model, device, train_loader, loss_function, 
            augmentation, alpha_max, cosine, temp, logger, "Train")
        val_loss = compute_similarity_loss(model, device, valid_loader, loss_function, 
            augmentation, alpha_max, cosine, temp, logger, "Validation")

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

        if optimizer.param_groups[0]['lr'] == plateau_factor * lr:
            change_epochs.append(epoch)
            lr = plateau_factor * lr
            logger.log("Learning rate decreasing to {}\n".format(lr))

    train_report(train_losses, val_losses, chnage_epochs=change_epochs, logger=logger)

    return train_losses, val_losses

#Epoch of supervised training
def train_sup_epoch(model, device, train_loader, loss_function, optimizer, epoch, log_interval, logger):
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
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            logger.log(log_str)

#Epoch of disatillation training (similarity-preserving)
def train_distillation_epoch(student, teacher, device, train_loader, loss_function, 
    optimizer, epoch, log_interval, logger, cosine):
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
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            logger.log(log_str)

#Epoch of similarity training
def train_similarity_epoch(model, device, train_loader, batch_size, loss_function, 
    optimizer, epoch, log_interval, logger, augmentation, alpha_max, beta, cosine, temp):

    '''
    DOCSTRING
    '''

    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        augmented_data, sim_prob = augment(data, augmentation, alpha_max, beta, device=device)
        output = get_model_similarity(model, data, augmented_data, cosine)
        if isinstance(loss_function, nn.KLDivLoss):
            #KL Divergence expects output to be log probability distribution
            eps = 1e-7
            output_comp = 1 - output
            output = torch.stack((output, output_comp), dim=1)
            sim_prob_comp = 1 - sim_prob
            sim_prob = torch.stack((sim_prob, sim_prob_comp), dim=1)
            loss = loss_function((output+eps).log(), sim_prob.detach())
        else:
            loss = loss_function(output, sim_prob.detach())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            log_str = 'Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            logger.log(log_str)

#Evaluation on data
def predict(model, device, loader, loss_function, logger, subset):
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

#Takes an "Embedder" model as input
#Return embeddings in numpy format
def get_embeddings(model, device, loader, emb_dim):
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

#Extract the labels as a numpy array
def get_labels(loader):
    labels = np.zeros((0))
    for _, target in loader:
        labels = np.concatenate((labels, target))

    return labels

#Get similarities between instances, as given by student and teacher
def get_student_teacher_similarity(student, teacher, data, cosine):
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
    

#Get normalized similarities between original and augmented data, as predicted by the model 
def get_model_similarity(model, data, augmented_data, cosine, temp=1):
    #Get embeddings of original data
    data_embs = model(data)
    #Get embeddings of augmented data
    augmented_data_embs = model(augmented_data)
    if cosine: #Using cosine similarity
        data_embs = F.normalize(data_embs, p=2, dim=1)
        augmented_data_embs = F.normalize(augmented_data_embs, p=2, dim=1)
    model_sims = torch.sum(torch.mul(data_embs, augmented_data_embs), dim=1)
    if cosine:
        output = model_sims
    else: #Use tempered sigmoid to map scores to (0,1)
        output = 1 / (1 + torch.exp(-temp * model_sims))

    return output

#Compute loss over the entire set
def compute_distillation_loss(student, teacher, device, loader, loss_function, 
    cosine, logger, subset):
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

def compute_similarity_loss(model, device, loader, loss_function, augmentation, alpha_max, 
    cosine, temp, logger, subset):
    model.eval()
    losses = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            augmented_data, sim_prob = augment(data, augmentation, alpha_max, device=device)
            output = get_model_similarity(model, data, augmented_data, cosine, temp)
            if isinstance(loss_function, nn.KLDivLoss):
                #KL Divergence expects output to be log probability distribution
                eps = 1e-7
                output_comp = 1 - output
                output = torch.stack((output, output_comp), dim=1)
                sim_prob_comp = 1 - sim_prob
                sim_prob = torch.stack((sim_prob, sim_prob_comp), dim=1)
                losses.append(loss_function((output+eps).log(), sim_prob.detach()).item())
            else:
                losses.append(loss_function(output, sim_prob.detach()).item())

    loss = np.mean(losses)

    log_str = '\n{} set: Average loss: {:.6f}\n'.format(subset, loss)
    logger.log(log_str)

    return loss
            
def train_report(train_losses, val_losses, train_accs=None, val_accs=None, change_epochs=None, logger=None):
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

#Helper function for plots of accuracy vs. epochs and loss vs. epochs
def train_plots(train_vals, val_vals, y_label, save_dir, change_epochs):
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