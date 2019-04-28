import torch
import numpy as np
import os


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        if input.requires_grad:
            input = input.detach()
        return input.cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def module_size(module):
    assert isinstance(module, torch.nn.Module)
    n_params, n_conv_layers = 0, 0
    for name, param in module.named_parameters():
        if 'conv' in name or 'Conv' in name:
            n_conv_layers += 1
        n_params += param.numel()
    return n_params, n_conv_layers


def stitch_pathes(four_crops):
    """for particular use, each one of them is 256x256, stitch to 512x512
    from torchvision `five_crop`
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))

    Args:
        four_crops: (4, 1, 256, 256) numpy array
    
    Returns:
        big_image (1, 512, 512)
    """
    crop_h, crop_w = four_crops.shape[-2], four_crops.shape[-1]
    
    stitched = np.zeros((four_crops.shape[1], crop_h*2, crop_w*2))
    stitched[:, 0:crop_h, 0:crop_w] = four_crops[0]
    stitched[:, 0:crop_h, crop_w:] = four_crops[1]
    stitched[:, crop_h:, 0:crop_w] = four_crops[2]
    stitched[:, crop_h:, crop_w:] = four_crops[3]

    return stitched



