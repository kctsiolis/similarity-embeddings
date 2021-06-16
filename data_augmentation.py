from torchvision import transforms
import torch
import numpy as np

#Gaussian blur with parameter sigma (which controls the blur intensity)
def gaussian_blur_sigma(data, sigma_max, device='cpu'):
    if sigma_max <= 0:
        raise ValueError("Maximum standard deviation must be positive.")
    sigma = (torch.rand(data.shape[0])*sigma_max).to(device)
    img_size = min(data.shape[2], data.shape[3])
    #Kernel size must be odd
    kernel_size = img_size if img_size % 2 == 1 else img_size - 1 
    for i, image in enumerate(data):
        data[i,:,:,:] = transforms.GaussianBlur(kernel_size, sigma[i].item())(image)

    sim_prob = 1 - (1/sigma_max)*sigma

    return data, sim_prob

def gaussian_blur_kernel(data, max_kernel_size=None, device='cpu'):
    img_size = min(data.shape[2], data.shape[3])
    if max_kernel_size is None:
        max_kernel_size = img_size if img_size % 2 == 1 else img_size - 1 
    elif max_kernel_size % 2 == 0:
        raise ValueError("Maximum kernel size cannot be odd.")
    elif max_kernel_size < 1:
        raise ValueError("Maximum kernel size must be at least 1.")
    #Set the possible kernel sizes (must be odd numbers)
    kernel_sizes = np.arange(max_kernel_size // 2 + 1) * 2 + 1
    #Sample kernel sizes (one per image in the batch)
    alpha = np.random.choice(kernel_sizes, size=(data.shape[0],))
    alpha = torch.from_numpy(alpha).to(device)
    for i, image in enumerate(data):
        data[i,:,:,:] = transforms.GaussianBlur(alpha[i].item(), 1)(image)

    sim_prob = (1/(1-max_kernel_size))*(alpha - 1) + 1

    return data, sim_prob

#Main augmentation function
#Input: 
#   data: Tensor of image data (Batch size x channels x height x width)
#   augmentation: Type of augmentation
#   alpha_max: Largest possible augmentation strength
#Output:
#   augmented_data: The augmented batch
#   sim_prob: Tensor of similarity probabilities (computed based on augmentation strength)
def augment(data, augmentation, alpha_max=None, device='cpu'):
    #Sample an augmentation strength for each instance in the batch
    augmented_data = data.detach().clone()
    if augmentation == 'blur-sigma':
        return gaussian_blur_sigma(augmented_data, alpha_max, device)
    elif augmentation == 'blur-kernel':
        return gaussian_blur_kernel(augmented_data, alpha_max, device)
    else:
        raise NotImplementedError
