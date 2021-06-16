#Useful functions for training neural nets
#Based on code from PyTorch example
#https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from data_augmentation import augment

#Supervised training loop
def train_sup(model, train_loader, valid_loader, device='cpu', seed=42, train_batch_size=64, 
    valid_batch_size=1000, loss_function=nn.CrossEntropyLoss, epochs=20, lr=0.1,
    step_size=5, gamma=0.1, early_stop=5, log_interval=10, save_path=None, plots_dir=None):
    
    torch.manual_seed(seed)
    device = torch.device(device)
    print("Running on device {}".format(device))
    model = model.to(device)

    #TODO: Generalize this to any optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    #Keep track of losses and accuracies
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs_until_stop = early_stop

    for epoch in range(1, epochs + 1):
        train_sup_epoch(model, device, train_loader, loss_function, optimizer, epoch, log_interval)
        train_loss, train_acc = predict(model, device, train_loader, loss_function, "Train")
        val_loss, val_acc = predict(model, device, valid_loader, loss_function, "Validation")

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
                print("Saving model...")
                torch.save(model.state_dict(), save_path)
                print("Model saved.\n")

        scheduler.step()

    train_report(train_losses, val_losses, train_accs, val_accs, plots_dir)

    return train_losses, train_accs, val_losses, val_accs

def train_distillation(student, teacher, train_loader, valid_loader, device='cpu', 
    seed=42, train_batch_size=64, valid_batch_size=1000, loss_function=nn.MSELoss, epochs=20, 
    lr=0.1, step_size=5, gamma=0.1, early_stop=5, log_interval=10, save_path=None, plots_dir=None, 
    cosine=False):

    torch.manual_seed(seed)
    device = torch.device(device)
    print("Running on device {}".format(device))
    student = student.to(device)
    teacher = teacher.to(device)

    #TODO: Generalize this to any optimizer
    optimizer = optim.Adam(student.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    #Keep track of losses and accuracies
    train_losses = []
    val_losses = []
    epochs_until_stop = early_stop

    for epoch in range(1, epochs + 1):
        train_distillation_epoch(student, teacher, device, train_loader, loss_function, 
            optimizer, epoch, log_interval, cosine)
        train_loss = compute_distillation_loss(student, teacher, device, train_loader, 
            loss_function, cosine, "Training")
        val_loss = compute_distillation_loss(student, teacher, device, valid_loader, 
            loss_function, cosine, "Validation")

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
                print("Saving model...")
                torch.save(student.state_dict(), save_path)
                print("Model saved.\n")

        scheduler.step()

    train_report(train_losses, val_losses, save_dir=plots_dir)

    return train_losses, val_losses

#Supervised training loop
def train_similarity(model, train_loader, valid_loader, device='cpu', augmentation='blur-sigma',
    alpha_max=10, seed=42, train_batch_size=64, valid_batch_size=1000, loss_function=nn.MSELoss, epochs=50, 
    lr=0.1, step_size=5, gamma=0.1, early_stop=5, log_interval=10, save_path=None, plots_dir=None, cosine=False):
    
    torch.manual_seed(seed)
    device = torch.device(device)
    print("Running on device {}".format(device))
    model = model.to(device)

    #TODO: Generalize this to any optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    #Keep track of losses and accuracies
    train_losses = []
    val_losses = []
    epochs_until_stop = early_stop

    for epoch in range(1, epochs + 1):
        train_similarity_epoch(model, device, train_loader, train_batch_size, loss_function, 
            optimizer, epoch, log_interval, augmentation, alpha_max, cosine)
        train_loss = compute_similarity_loss(model, device, train_loader, loss_function, 
            augmentation, alpha_max, cosine, "Train")
        val_loss = compute_similarity_loss(model, device, valid_loader, loss_function, 
            augmentation, alpha_max, cosine, "Validation")

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
                print("Saving model...")
                torch.save(model.state_dict(), save_path)
                print("Model saved.\n")

        scheduler.step()

    train_report(train_losses, val_losses, save_dir=plots_dir)

    return train_losses, val_losses

#Epoch of supervised training
def train_sup_epoch(model, device, train_loader, loss_function, optimizer, epoch, log_interval):
    model.train()
    loss_function = loss_function() #Instantiate loss
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            )

#Epoch of disatillation training (similarity-preserving)
def train_distillation_epoch(student, teacher, device, train_loader, loss_function, 
    optimizer, epoch, log_interval, cosine):
    student.train()
    teacher.eval()
    loss_fn = loss_function()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        student_sims, teacher_sims = get_student_teacher_similarity(student, teacher, data, cosine)
        loss = loss_fn(student_sims, teacher_sims)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            )

#Epoch of similarity training
def train_similarity_epoch(model, device, train_loader, batch_size,
    loss_function, optimizer, epoch, log_interval, augmentation, alpha_max, cosine):
    model.train()
    loss_function = loss_function() #Instantiate loss
    if not cosine:
        #Use the sigmoid function to map outputs to (0,1)
        sigmoid = nn.Sigmoid()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        augmented_data, sim_prob = augment(data, augmentation, alpha_max, device=device)
        output = get_model_similarity(model, data, augmented_data, cosine)
        loss = loss_function(output, sim_prob.detach())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            )

#Evaluation on data
def predict(model, device, loader, loss_function, subset):
    model.eval()
    loss_function = loss_function(reduction='sum')
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    loss /= len(loader.dataset)

    acc = 100.00 * correct/ len(loader.dataset)

    print('\n{} set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(subset,
        loss, correct, len(loader.dataset), acc
    ))

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
def get_model_similarity(model, data, augmented_data, cosine):
    #Get embeddings of original data
    data_embs = model(data)
    #Get embeddings of augmented data
    augmented_data_embs = model(augmented_data)
    if cosine: #Using cosine similarity
        data_embs = F.normalize(data_embs, p=2, dim=1)
        augmented_data_embs = F.normalize(augmented_data_embs, p=2, dim=1)
    model_sims = torch.sum(torch.mul(data_embs, augmented_data_embs), dim=1)
    if cosine:
        #Cosine similarity will already be between 0 and 1 due to non-negativity of embeddings
        output = model_sims
    else: #Use sigmoid to map similarities to (0,1)
        sigmoid = nn.Sigmoid()
        output = sigmoid(model_sims)

    return output

#Compute loss over the entire set
def compute_distillation_loss(student, teacher, device, loader, loss_function, cosine, subset):
    student.eval()
    teacher.eval()
    loss_function = loss_function()
    losses = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            student_sims, teacher_sims = get_student_teacher_similarity(student, teacher, data, cosine)
            losses.append(loss_function(student_sims, teacher_sims).item())

    loss = np.mean(losses)

    print('\n{} set: Average loss: {:.6f}\n'.format(subset, loss))

    return loss

def compute_similarity_loss(model, device, loader, loss_function, augmentation, alpha_max, cosine, subset):
    model.eval()
    loss_function = loss_function()
    losses = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            augmented_data, sim_prob = augment(data, augmentation, alpha_max, device=device)
            output = get_model_similarity(model, data, augmented_data, cosine)
            losses.append(loss_function(output, sim_prob).item())

    loss = np.mean(losses)

    print('\n{} set: Average loss: {:.6f}\n'.format(subset, loss))

    return loss
            

def train_report(train_losses, val_losses, train_accs=None, val_accs=None, save_dir=None):
    best_epoch = np.argmin(val_losses)
    print("Training complete.\n")
    print("Best Epoch: {}".format(best_epoch + 1))
    print("Training Loss: {:.6f}".format(train_losses[best_epoch]))
    if train_accs is not None:
        print("Training Accuracy: {:.2f}".format(train_accs[best_epoch]))
    print("Validation Loss: {:.6f}".format(val_losses[best_epoch]))
    if val_accs is not None:
        print("Validation Accuracy: {:.2f}".format(val_accs[best_epoch]))

    #Save loss and accuracy plots
    if save_dir is not None:
        train_plots(train_losses, val_losses, "Loss", save_dir)
        if train_accs is not None and val_accs is not None:
            train_plots(train_accs, val_accs, "Accuracy", save_dir)

#Helper function for plots of accuracy vs. epochs and loss vs. epochs
def train_plots(train_vals, val_vals, y_label, save_dir):
    epochs = np.arange(1,len(train_vals)+1)
    plt.figure()
    plt.plot(epochs, train_vals, 'b-')
    plt.plot(epochs, val_vals, 'r-')
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.savefig(save_dir + '/' + y_label + '_plots')