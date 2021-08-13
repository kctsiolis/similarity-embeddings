"""Code for data augmentation on images.

Augmentations supported:
    Gaussian blur 
    Colour jitter
    Random crop

"""

from torchvision import transforms
import torch
from torch.distributions import Categorical

class Augmentation():
    """Generic augmentation class. Parent class of the augmentations.
    
    Attributes:
        alpha_min: Minimum augmentation strength.
        alpha_max: Maximum augmentation strength.
        device: Device to run on.
        random: Whether or not to sample augmentation strengths randomly.
        simclr: Whether or not to use SimCLR's sampling regime. If it is set to
            true, then alpha_min, alpha_max, and random will be ignored.

    """
    def __init__(
            self, alpha_min: float = 0, alpha_max: float = 1, 
            device: torch.device = 'cpu', random: bool = True,
            simclr: bool = False):
        if alpha_min < 0 or alpha_max > 1:
            raise ValueError('The range of alpha must be a subset of [0,1].')
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.device = device
        self.random = random
        self.simclr = simclr

    def augment(self, data: torch.Tensor):
        """Perform data augmentation.
        
        Args:
            data: The input to perform the augmentation on.

        Returns:
            The augmented data and the assoicated augmentation strengths.

        """
        batch_size = data.shape[0]
        if self.random:
            alpha = (torch.rand(
                batch_size)*(self.alpha_max-self.alpha_min) + 
                self.alpha_min).to(self.device)
        else:
            alpha = (torch.ones(batch_size)*self.alpha_max).to(self.device)

        if self.simclr:
            return self.apply_simclr_transform(data, alpha)
        else:
            return self.apply_transform(data, alpha)

    def apply_transform(self, data: torch.Tensor, alpha: torch.Tensor):
        pass

    def apply_simclr_transform(self, data: torch.Tensor):
        pass

class GaussianBlur(Augmentation):

    def __init__(
            self, alpha_min: float = 0, alpha_max: float = 1, 
            kernel_size: int = None, device: torch.device = 'cpu',
            random: bool = True, simclr: bool = False):
        super().__init__(alpha_min, alpha_max, device, random, simclr)
        if self.simclr:
            self.alpha_min = 0.1
            self.alpha_max = 2.0
            self.kernel_size = None
        else:
            self.kernel_size = kernel_size

    def apply_transform(self, data: torch.Tensor, alpha: torch.Tensor):
        augmented_data = data.clone()
        if self.kernel_size is None:
            img_size = min(data.shape[2], data.shape[3])
            self.kernel_size = img_size if img_size % 2 == 1 else img_size - 1 

        sigma = alpha * 5
        for i, image in enumerate(data):
            augmented_data[i,:,:,:] = transforms.GaussianBlur(self.kernel_size, sigma[i].item())(image)

        return augmented_data, alpha

    def apply_simclr_transform(self, data: torch.Tensor, alpha: torch.Tensor):
        augmented_data = data.clone()

        if self.kernel_size is None:
            img_size = min(data.shape[2], data.shape[3])
            k = img_size // 10
            self.kernel_size = k if k % 2 == 1 else k - 1

        for i, image in enumerate(data):
            augmented_data[i,:,:,:] = transforms.GaussianBlur(self.kernel_size, alpha[i].item())(image)

        return augmented_data, alpha

class ColorJitter(Augmentation):
    def __init__(
            self, alpha_min: float = 0, alpha_max: float = 1, 
            device: torch.device = 'cpu', random: bool = True, 
            simclr: bool = False):
        super().__init__(alpha_min, alpha_max, device, random, simclr)

    def apply_transform(self, data: torch.Tensor, alpha: torch.Tensor):
        augmented_data = data.clone()
        # m = RelaxedOneHotCategorical(torch.Tensor([1.0]), torch.Tensor([
        #     0.25, 0.25, 0.25, 0.25]))
        # m_sample = m.sample(sample_shape=torch.Size([num_samples]))
        pm = Categorical(torch.Tensor[0.5, 0.5])
        pm_sample = pm.sample(sample_shape=torch.Size([data.shape[0], 3])) * 2 - 1
        for i, image in enumerate(data):
            # b, c, s, h = m_sample[i][0], m_sample[i][1], m_sample[i][2], m_sample[i][3]
            # b, c, s, h = b*a[i].item(), c*a[i].item(), s*a[i].item(), (h*a[i].item())/2
            b = 1 + pm_sample[i][0]*alpha[i].item()
            c = 1 + pm_sample[i][1]*alpha[i].item()
            s = 1 + pm_sample[i][2]*alpha[i].item()
            h = alpha[i].item() / 2
            augmented_data[i,:,:,:] = transforms.ColorJitter((b,b), (c,c), (s,s), (h,h))(image)

        return augmented_data, alpha

    def apply_simclr_transform(self, data: torch.Tensor, alpha: torch.Tensor):
        augmented_data = data.clone()
        alpha = torch.ones((data.shape[0]))
        for i, image in enumerate(data):
            augmented_data[i,:,:,:] = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)(image)

        return augmented_data, alpha

class RandomCrop(Augmentation):
    def __init__(
            self, alpha_min: float = 0, alpha_max: float = 1, 
            device: torch.device = 'cpu', random: bool = True, 
            simclr: bool = False):
        super().__init__(alpha_min, alpha_max, device, random, simclr)
        if simclr:
            self.alpha_min = 0.08
            self.alpha_max = 1.0

    def apply_transform(self, data: torch.Tensor, alpha: torch.Tensor):
        augmented_data = data.clone()
        for i, image in enumerate(data):
            sz = 1 - alpha[i].item()
            augmented_data[i,:,:,:] = transforms.RandomResizedCrop(size=image.shape[-2:], scale=(sz,sz))(image)

        return augmented_data, alpha

    def apply_simclr_transform(self, data: torch.Tensor, alpha: torch.Tensor):
        return self.apply_transform(data, alpha)

def make_augmentation(
        augmentation: str, alpha_min: float = 0, alpha_max: float = 1, 
        kernel_size: int = 223, device: torch.device = 'cpu', 
        random: bool = True, simclr: bool = False):
    """Instantiate an augmentation."""
    if augmentation == 'blur':
        return GaussianBlur(
            alpha_min, alpha_max, kernel_size, device, random, simclr
        )
    elif augmentation == 'jitter':
        return ColorJitter(
            alpha_min, alpha_max, device, random, simclr
        )
    else:
        return RandomCrop(
            alpha_min, alpha_max, device, random, simclr
        )
