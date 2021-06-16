from torchvision import transforms
import torch

#Gaussian blur with parameter sigma (standard deviation)
#Input: 
#   data: Tensor of image data
#   alpha: Array of blur intensities (one per image)
def gaussian_blur(data, alpha, kernel_size=27, device='cpu'):
    augmented_data = data.detach().clone().to(device)
    for i, image in enumerate(augmented_data):
        augmented_data[i,:,:,:] = transforms.GaussianBlur(kernel_size, alpha[i].item())(image)

    return augmented_data

#Probability of similarity for parameter alpha
def sim_prob(alpha, augmentation='blur'):
    if augmentation == 'blur':
        return 1-0.2*alpha
    else:
        raise NotImplementedError
