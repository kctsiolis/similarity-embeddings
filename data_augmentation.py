from torchvision import transforms
import torch
import numpy as np

#Gaussian blur with parameter sigma (which controls the blur intensity)
def gaussian_blur_sigma(data, sigma_max, b=0.2, device='cpu'):
    if sigma_max <= 0:
        raise ValueError("Maximum standard deviation must be positive.")
    sigma = (torch.rand(data.shape[0])*sigma_max).to(device)
    img_size = min(data.shape[2], data.shape[3])
    #Kernel size must be odd
    kernel_size = img_size if img_size % 2 == 1 else img_size - 1 
    for i, image in enumerate(data):
        data[i,:,:,:] = transforms.GaussianBlur(kernel_size, sigma[i].item())(image)

    sim_prob = torch.exp(-b * sigma)

    return data, sim_prob

def gaussian_blur_kernel(data, sigma=8, b=0.2, device='cpu'):
    img_size = min(data.shape[2], data.shape[3])
    max_kernel_size = img_size if img_size % 2 == 1 else img_size - 1 
    #Set the possible kernel sizes (must be odd numbers)
    kernel_sizes = np.arange(max_kernel_size // 2 + 1) * 2 + 1
    #Sample kernel sizes (one per image in the batch)
    alpha = np.random.choice(kernel_sizes, size=(data.shape[0],))
    alpha = torch.from_numpy(alpha).to(device)
    for i, image in enumerate(data):
        data[i,:,:,:] = transforms.GaussianBlur(alpha[i].item(), sigma)(image)

    #Compute similarity probability
    #Map kernel sizes to [0, kernel_size // 2 + 1]
    #Add noise in order to get continuity
    alpha = (alpha - 1) / 2
    sim_prob = torch.exp(-b * (alpha + torch.rand(data.shape[0]).to(device) - 0.5))

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
        return gaussian_blur_sigma(augmented_data, alpha_max, device=device)
    elif augmentation == 'blur-kernel':
        return gaussian_blur_kernel(augmented_data, device=device)
    else:
        raise NotImplementedError
