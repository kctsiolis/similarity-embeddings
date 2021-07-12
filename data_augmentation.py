"""Code for data augmentation on images.

Augmentations supported:
    Gaussian blur 

"""

from torchvision import transforms
import torch
import numpy as np

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

def gaussian_blur_kernel(data: torch.Tensor, device: torch.device, 
    sigma: float = 8.0, beta: float = 0.2) -> tuple([torch.Tensor, torch.Tensor]):
    """Apply gaussian blur with randomly sampled kernel size parameter.

    Kernel size is sampled uniformly i.i.d. (over valid values) for 
    each image in the data, while the standard deviation is fixed.

    Args:
        data: The data to be augmented.
        device: The device to store the data on for augmentation.
        sigma: The standard deviation of the Gaussian blur.
        beta: Parameter of the similarity probability.

    Returns:
        The augmented data and the similarity probabilities

    """
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
    sim_prob = torch.exp(-beta * (alpha + torch.rand(data.shape[0]).to(device) - 0.5))

    return data, sim_prob

def augment(data: torch.Tensor, augmentation: str, device: torch.device, 
    alpha_max: float = 15.0, beta: float = 0.2, random: bool = True
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
    elif augmentation == 'blur-kernel':
        return gaussian_blur_kernel(augmented_data, device)
    else:
        raise NotImplementedError
