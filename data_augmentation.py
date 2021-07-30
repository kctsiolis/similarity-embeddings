"""Code for data augmentation on images.

Augmentations supported:
    Gaussian blur 
    Colour jitter
    Random crop

"""

from torchvision import transforms
import torch
from torch.distributions import Categorical, RelaxedOneHotCategorical

def gaussian_blur(data: torch.Tensor, alpha_max: float, 
    beta: float, device = torch.device, random: bool = True
    ) -> tuple([torch.Tensor, torch.Tensor]):
    """Apply gaussian blur with randomly sampled sigma parameter.

    Sigma is sampled uniformly i.i.d. for each image in the data,
    while the kernel size is fixed to be as large as possible.

    Args:
        data: Input data to be augmented.
        alpha_max: Largest possible augmentation intensity.
        beta: Similiarity probability parameter.
        device: Device on which to store batch.
        random: If true, sample intensities i.i.d., otherwise use sigma_max.

    Returns:
        The augmented data and the similarity probabilities.

    """
    if alpha_max <= 0:
        raise ValueError("Maximum standard deviation must be positive.")

    sigma_max = alpha_max * 5

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

def color_jitter(data: torch.Tensor, alpha_max: float, 
    beta: float, device = torch.device, random: bool = True
    ) -> tuple([torch.Tensor, torch.Tensor]):
    """Apply colour jitter data augmentation.

    Since colour jitter has four parameters (brightness, contrast,
    saturation, and hue), the augmentation strength alpha represents
    the sum of these. We sample from the relaxed categorical distribution
    to weigh each of the four parameters and then multiply by alpha.

    Args:
        data: Input data to be augmented.
        alpha_max: Maximum augmentation strength.
        beta: Similarity probability parameter.
        device: Device on which to store batch.
        random: If true, sample intensities i.i.d. Otherwise, use alpha_max.

    Returns:
        The augmented data with the similarity probabilities.
    
    """

    num_samples = data.shape[0]
    if alpha_max < 0 or alpha_max > 1:
        raise ValueError("Maximum augmentation strength must be in [0,1].")
    if random:
        a = (torch.rand(num_samples)*alpha_max).to(device)
    else:
        a = (torch.ones(num_samples)*alpha_max).to(device)

    # m = RelaxedOneHotCategorical(torch.Tensor([1.0]), torch.Tensor([
    #     0.25, 0.25, 0.25, 0.25]))
    # m_sample = m.sample(sample_shape=torch.Size([num_samples]))

    pm = Categorical(torch.Tensor([0.5, 0.5]))
    pm_sample = pm.sample(sample_shape=torch.Size([num_samples, 3])) * 2 - 1

    for i, image in enumerate(data):
        # b, c, s, h = m_sample[i][0], m_sample[i][1], m_sample[i][2], m_sample[i][3]
        # b, c, s, h = b*a[i].item(), c*a[i].item(), s*a[i].item(), (h*a[i].item())/2
        b = 1 + pm_sample[i][0]*a[i].item()
        c = 1 + pm_sample[i][1]*a[i].item()
        s = 1 + pm_sample[i][2]*a[i].item()
        h = a[i].item() / 2
        data[i,:,:,:] = transforms.ColorJitter((b,b), (c,c), (s,s), (h,h))(image)

    sim_prob = torch.exp(-beta * a)

    return data, sim_prob

def random_crop(data: torch.Tensor, alpha_max: int,
    beta: float, device = torch.device, random: bool = True
    ) -> tuple([torch.Tensor, torch.Tensor]):
    """Perform random crop augmentation on the input.
    
    Here, alpha controls the proportion of the image that is thrown
    away in the crop. For example, alpha = 0 retains the entire image,
    while alpha = 0.5 retains half the image. In all cases, a random
    portion of the image is chosen for the crop.

    Args:
        data: Input data to be augmented.
        alpha_max: Maximum augmentation strength.
        beta: Similarity probability parameter.
        device: Device on which to store batch.
        random: If true, sample intensities i.i.d. Otherwise, use alpha_max.

    Returns:
        The augmented data with the similarity probabilities.

    """
    img_size = min(data.shape[2], data.shape[3])
    if alpha_max >= 1 or alpha_max < 0:
        raise ValueError("Maximum augmentation strength must be in [0,1).")
    if random:
        s = (torch.rand(data.shape[0])*alpha_max).to(device)
    else:
        s = (torch.ones(data.shape[0])*alpha_max).to(device)
    
    for i, image in enumerate(data):
        sz = 1 - s[i].item()
        data[i,:,:,:] = transforms.RandomResizedCrop(size=image.shape[-2:], scale=(sz,sz))(image)

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
        NotImplementedError: Only Gaussian blur, colour jitter, and random crop are implemented.

    """
    #Sample an augmentation strength for each instance in the batch
    augmented_data = data.detach().clone()
    if augmentation == 'blur':
        return gaussian_blur(augmented_data, alpha_max, beta, device, random)
    elif augmentation == 'color-jitter':
        return color_jitter(augmented_data, alpha_max, beta, device, random)
    elif augmentation == 'random-crop':
        return random_crop(augmented_data, alpha_max, beta, device, random)
    else:
        raise NotImplementedError
