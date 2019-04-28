import torch
import numpy as np
import torch.nn.functional as F
from skimage.measure import compare_psnr, compare_ssim
from utils.misc import to_numpy

def cal_psnr(clean, noisy, max_val=255, normalized=True):
    """
    Args:
        clean (Tensor): [0, 255], BCHW
        noisy (Tensor): [0, 255], BCHW
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        PSNR per image: (B,)
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)
    mse = F.mse_loss(noisy, clean, reduction='none').view(clean.shape[0], -1).mean(1)
    return 10 * torch.log10(max_val ** 2 / mse)


def cal_ssim(clean, noisy, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM

    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)

    clean, noisy = to_numpy(clean), to_numpy(noisy)
    ssim = np.array([compare_ssim(clean[i, 0], noisy[i, 0], data_range=255) 
        for i in range(clean.shape[0])])

    return ssim   


def cal_psnr2(clean, noisy, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM

    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)

    clean, noisy = to_numpy(clean), to_numpy(noisy)

    psnr = np.array([compare_psnr(clean[i, 0], noisy[i, 0], data_range=255) 
        for i in range(clean.shape[0])])

    return psnr   





    



