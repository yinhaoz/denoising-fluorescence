import torch
import numpy as np
from time import time


def add_noise(image, mode='poisson', psnr=25, noisy_per_clean=2, clip=False):
    """This function is called to create noisy images, after getting minibatch
    of clean images from dataloader. 
    Works well for non-saturating images, uint8.
    
    Different implementation of scaling of pixel values for the mean of 
    Poisson noise.

    References:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/util/noise.py
        https://www.mathworks.com/help/images/ref/imnoise.html#mw_226e1fb2-f53a-4e49-9bb1-6b167fc2eac1
        http://reference.wolfram.com/language/ref/ImageEffect.html
        https://imagej.nih.gov/ij/plugins/poisson-noise.html

    Args:
        image (torch.Tensor): image after `torchvision.transforms.ToTensor`, 
            range [0.0, 1.0], (B, C, H, W)
        mode (str): Default `Poisson`, other kinds of noise on the way... 
        psnr (float): Peak-SNR in DB. If it is list of size 2, then uniformly 
            select one psnr from this range
        noisy_per_clean (int): return number of noisy images per clean image
        clip (bool): clip the noisy output or not
    """
    
    if mode == 'poisson':
        if image.dtype == torch.uint8:
            max_val = 255
        elif image.dtype == torch.int16:
            max_val = 32767 if image.max() > 4095 else 4095
        else:
            raise TypeError('image data type is expected to be either uint8 '\
                'or int16, but got {}'.format(image.dtype)) 
        if noisy_per_clean > 1:
            image = image.repeat(noisy_per_clean, 1, 1, 1)
        image = image.float()

        if isinstance(psnr, (list, tuple)):
            assert len(psnr) == 2, 'please specify the range of PSNR using '\
                'only two numbers'
            # randomly select noise level for each channel
            psnr = torch.randn(image.shape[0]).uniform_(psnr[0], psnr[1]).to(image.device)
        scale = 10 ** (psnr / 10) * image.view(image.size(0), -1).mean(1) / max_val ** 2
        scale = scale.view(image.size(0), 1, 1, 1)
        noisy = torch.poisson(image * scale) / scale
        return torch.clamp(noisy, 0., max_val) if clip else noisy

    else:
        raise NotImplementedError('Other noise mode to be implemented')
