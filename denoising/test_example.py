"""Reproduce denoising example in Fig 6 & 7, Table 4 in the paper.

noise_levels = [1]
image_types = ['Confocal_BPAE_R', 'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH']

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
from PIL import Image
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
parser.add_argument('--pretrain-dir', default='./experiments/pretrained', type=str, help='dir to pre-trained model')
parser.add_argument('--data-root', default='./dataset', type=str, help='dir to dataset')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use GPU or not, default using GPU')
args_test = parser.parse_args()

test_batch_size = 1
test_seed = 13
cmap = 'inferno'
device = 'cpu' if args_test.no_cuda else 'cuda'

noise_levels = [1]
image_types = ['Confocal_BPAE_R', 'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH']

data_dir = args_test.data_root
run_dir = args_test.pretrain_dir + f'/{args_test.model}'

with open(run_dir + '/args.txt') as args_file:
    args = Namespace(**json.load(args_file))
pprint(args)
if args_test.no_cuda:
    test_dir = run_dir + '/example_cpu'
else:
    test_dir = run_dir + '/example_gpu'
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
    psnr_3c = 0.
    ssim_3c = 0.
    for i, image_type in enumerate(image_types):
        tic = time.time()
        if image_type != 'Confocal_FISH':
            noisy_file = data_dir + f'/{image_type}/raw/19/HV110_P0500510000.png'
        else:
            noisy_file = data_dir + f'/{image_type}/raw/19/HV140_P100510000.png'
        clean_file = data_dir + f'/{image_type}/gt/19/avg50.png'
        noisy = four_crop(Image.open(noisy_file)).to(device)
        clean = four_crop(Image.open(clean_file)).to(device)
        print(noisy.shape)
        print(clean.shape)
        
        denoised = model(noisy.to(device))
        psnr = cal_psnr(clean, denoised).mean(0)
        ssim = cal_ssim(clean, denoised).mean(0)

        denoised = stitch_pathes(to_numpy(denoised))[0]
        noisy = stitch_pathes(to_numpy(noisy))[0]
        clean = stitch_pathes(to_numpy(clean))[0]

        print(image_type)
        print(f'psnr: {psnr}')
        print(f'ssim: {ssim}')
        if i < 3:
            psnr_3c += psnr.item()
            ssim_3c += ssim.item()
        save_file = test_dir + f'/{args_test.model}_noise{noise_level}_{image_type}_test19_idx0_denoised.png'
        plt.imsave(save_file, denoised, format="png", cmap="gray")
    print('Confocal BPAE avg of 3 channels')
    print(f'PSNR avg: {psnr_3c / 3.}')
    print(f'SSIM avg: {ssim_3c / 3.}')

print(f'Done in {time.time()-gtic} sec')
