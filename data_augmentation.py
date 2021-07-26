"""Code for data augmentation on images.

Augmentations supported:
    Gaussian blur 

"""

from torchvision import transforms
import torch
import math

#Gaussian blur with parameter sigma (which controls the blur intensity)
def gaussian_blur_sigma(data: torch.Tensor, sigma_max: float, 
    beta: float, device = torch.device, random: bool = True
    ) -> tuple([torch.Tensor, torch.Tensor]):
    """Apply gaussian blur with randomly sampled sigma parameter.

    Sigma is sampled uniformly i.i.d. for each image in the data,
    while the kernel size is fixed to be as large as possible.

    Args:
        data: Input data to be augmented.
        sigma_max: Largest possible augmentation intensity.
        beta: Similiarity probability parameter.
        device: Device on which to store batch.
        random: If true, sample intensities i.i.d., otherwise use sigma_max.

    Returns:
        The augmented data and the similarity probabilities.

    """
    if sigma_max <= 0:
        raise ValueError("Maximum standard deviation must be positive.")
    if random:
        sigma = (torch.rand(data.shape[0])*sigma_max).to(device)
    else:
        sigma = (torch.ones(data.shape[0])*sigma_max).to(device)
    img_size = min(data.shape[2], data.shape[3])
    #Kernel size must be odd
    kernel_size = img_size if img_size % 2 == 1 else img_size - 1 
    for i, image in enumerate(data):
        data[i,:,:,:] = transforms.GaussianBlur(kernel_size, sigma[i].item())(image)

    sim_prob = torch.exp(-beta * sigma)

    return data, sim_prob

def color_jitter(data: torch.Tensor, s_max: float, 
    beta: float, device = torch.device, random: bool = True
    ) -> tuple([torch.Tensor, torch.Tensor]):
    if s_max < 0:
        raise ValueError("Maximum augmentation strength should be non-negative.")
    if random:
        s = (torch.rand(data.shape[0])*s_max).to(device)
    else:
        s = (torch.ones(data.shape[0])*s_max).to(device)

    for i, image in enumerate(data):
        data[i,:,:,:] = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)(image)

    sim_prob = torch.exp(-beta * s)

    return data, sim_prob

def random_crop(data: torch.Tensor, s_max: int,
    beta: float, device = torch.device, random: bool = True
    ) -> tuple([torch.Tensor, torch.Tensor]):
    img_size = min(data.shape[2], data.shape[3])
    if s_max >= img_size:
        raise ValueError("Maximum augmentation strength must be at most \
            one less than image height/width.")
    if random:
        s = (math.ceil(torch.rand(data.shape[0])*s_max)).to(device)
    else:
        s = (torch.ones(data.shape[0])*s_max).to(device)
    
    for i, image in enumerate(data):
        data[i,:,:,:] = transforms.RandomCrop(img_size-s)(image)

    sim_prob = torch.exp(-beta * (s / img_size))

    return data, sim_prob

def augment(data: torch.Tensor, augmentation: str, device: torch.device, 
    alpha_max: float, beta: float, random: bool = True
    ) -> tuple([torch.Tensor, torch.Tensor]):
    """Perform data augmentation on input.

    Args:
        data: The input data to be augmented.
        augmentation: The augmentation to perform.
        device: The device to store data on during augmentation.
        alpha_max: Largest possible augmentation intensity.
        beta: Similarity probability parameter.
        random: Whether or not intensities are sampled i.i.d.

    Returns:
        The augmented data and the similarity probabilities.

    Raises:
        NotImplementedError: Only Gaussian blur is implemented.

    """
    #Sample an augmentation strength for each instance in the batch
    augmented_data = data.detach().clone()
    if augmentation == 'blur-sigma':
        return gaussian_blur_sigma(augmented_data, alpha_max, beta, device, random)
    elif augmentation == 'color-jitter':
        return color_jitter(augmented_data, alpha_max, beta, device, random)
    elif augmentation == 'random_crop':
        return random_crop(augmented_data, alpha_max, beta, device, random)
    else:
        raise NotImplementedError
