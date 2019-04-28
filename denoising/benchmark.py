"""Benchmark blind Poisson denoising with pretrained models.
Reproduce Table 2 in the paper.

    noise_levels = [1, 2, 4, 8, 16]
    image_types:
        - test_mix
        - type/group_19: 12 types

    metrics:
        - psnr
        - ssim
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.dncnn import DnCNN
from models.unet import UnetN2N
from utils.misc import mkdirs, stitch_pathes, to_numpy, module_size
from utils.plot import save_samples, save_stats, plot_row
from utils.metrics import cal_psnr, cal_psnr2, cal_ssim
from utils.data_loader import load_denoising, load_denoising_test_mix, fluore_to_tensor
import numpy as np
import argparse
from argparse import Namespace
import json
import random
import time
import sys
from pprint import pprint
import matplotlib.pyplot as plt
plt.switch_backend('agg')


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='dncnn', choices=['dncnn', 'n2n'], type=str, help='the model name')
parser.add_argument('-bs', '--batch-size', default=1, type=int, help='test batch size')
# parser.add_argument('--data-root', default='./dataset', type=str, help='dir to dataset')
parser.add_argument('--data-root', default='/afs/crc.nd.edu/user/y/yzhu10/data/denoising/dataset', type=str, help='dir to dataset')
parser.add_argument('--pretrain-dir', default='./experiments/pretrained', type=str, help='dir to pre-trained model')
parser.add_argument('--noise-levels', default=[1, 2, 4, 8, 16], type=str, help='dir to pre-trained model')
parser.add_argument('--image-types', default=None, type=str, help='image type')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use GPU or not, default using GPU')
args_test = parser.parse_args()


test_batch_size = 1
test_seed = 13
cmap = 'inferno'
device = 'cpu' if args_test.no_cuda else 'cuda'

noise_levels = args_test.noise_levels
if args_test.image_types is not None:
    image_types = args_test.image_types
    assert isinstance(image_types, (list, tuple))
else:
    image_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
                   'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
                   'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
                   'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B', 
                   'test_mix']
    image_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
                   'TwoPhoton_MICE']
# image_types = ['test_mix']  
run_dir = args_test.pretrain_dir + f'/{args_test.model}'

with open(run_dir + '/args.txt') as args_file:  
    args = Namespace(**json.load(args_file))
pprint(args)
if args_test.no_cuda:
    test_dir = run_dir + '/benchmark_cpu'
else:
    test_dir = run_dir + '/benchmark_gpu'
mkdirs(test_dir)

if args_test.model == 'dncnn':
    model = DnCNN(depth=args.depth, 
                n_channels=args.width, 
                image_channels=1, 
                use_bnorm=True, 
                kernel_size=3)
elif args_test.model == 'n2n':
    model = UnetN2N(args.in_channels, args.out_channels)

if args.debug:
    print(model)
    print(module_size(model))
model.load_state_dict(torch.load(run_dir + f'/checkpoints/model_epoch{args.epochs}.pth', 
    map_location='cpu'))
model = model.to(device)
model.eval()

logger = {}
# (tl, tr, bl, br, center) --> only select the first four
four_crop = transforms.Compose([
    transforms.FiveCrop(args.imsize),
    transforms.Lambda(lambda crops: torch.stack([
        fluore_to_tensor(crop) for crop in crops[:4]])),
    transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
    ])

gtic = time.time()
for noise_level in noise_levels:
    for image_type in image_types:
        test_case_dir = test_dir + f'/noise{noise_level}_{image_type}'
        mkdirs(test_case_dir)
        tic = time.time()
        if image_type == 'test_mix':
            n_plots = 12
            test_loader = load_denoising_test_mix(args_test.data_root, 
                batch_size=test_batch_size, noise_levels=[noise_level],
                transform=four_crop, target_transform=four_crop, 
                patch_size=args.imsize)
        else:
            n_plots = 2
            test_loader = load_denoising(args_test.data_root, train=False, 
                batch_size=test_batch_size, noise_levels=[noise_level], 
                types=[image_type], captures=50,
                transform=four_crop, target_transform=four_crop, 
                patch_size=args.imsize)

        # four crop
        multiplier = 4
        n_test_samples = len(test_loader.dataset) * multiplier

        np.random.seed(test_seed)
        fixed_idx = np.random.permutation(len(test_loader.dataset))[:n_plots]
        print(f'fixed test index: {fixed_idx}')

        # (n_plots, 4, 1, 256, 256)
        fixed_test_noisy = torch.stack([(test_loader.dataset[i][0]) for i in fixed_idx])
        fixed_test_clean = torch.stack([(test_loader.dataset[i][1]) for i in fixed_idx])
        print(f'fixed test noisy shape: {fixed_test_noisy.shape}')
        fixed_test_noisy = fixed_test_noisy.to(device)

        case = {'noise': noise_level,
                'type': image_type,
                'samples': n_test_samples,
                }
        pprint(case)
        print('Start testing............')

        psnr, psnr2, ssim, time_taken = 0., 0., 0., 0
        for batch_idx, (noisy, clean) in enumerate(test_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            # fuse batch and four crop
            noisy = noisy.view(-1, *noisy.shape[2:])
            clean = clean.view(-1, *clean.shape[2:])
            tic_i = time.time()
            denoised = model(noisy)
            time_taken += (time.time() - tic_i)
            psnr += cal_psnr(clean, denoised).sum().item()
            ssim += cal_ssim(clean, denoised).sum()

        # time per 512x512 (training image is 256x256)
        time_taken /= (n_test_samples / multiplier)
        psnr = psnr / n_test_samples
        ssim = ssim / n_test_samples

        result = {'psnr': psnr,
                  'ssim': ssim,
                  'time': time_taken}
        case.update(result)
        pprint(result)
        logger.update({f'noise{noise_level}_{image_type}': case})

        # fixed test: (n_plots, 4, 1, 256, 256)
        for i in range(n_plots):
            print(f'plot {i}-th denoising: [noisy, denoised, clean]')
            fixed_denoised = model(fixed_test_noisy[i])
            fixed_noisy_stitched = stitch_pathes(to_numpy(fixed_test_noisy[i]))
            fixed_denoised_stitched = stitch_pathes(to_numpy(fixed_denoised))
            fixed_clean_stitched = stitch_pathes(to_numpy(fixed_test_clean[i]))
            plot_row(np.concatenate((fixed_noisy_stitched, fixed_denoised_stitched, 
                fixed_clean_stitched)), test_case_dir, f'denoising{i}', 
                same_range=True, plot_fn='imshow', cmap=cmap, colorbar=False)

        with open(test_dir + "/results_{}.txt".format('cpu' if args_test.no_cuda else 'gpu'), 'w') as args_file:
            json.dump(logger, args_file, indent=4)
        print(f'done test in {time.time()-tic} seconds')

print(f'Finally done in {time.time()-gtic} sec')
