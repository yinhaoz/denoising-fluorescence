"""Training Noise2Noise model
https://github.com/NVlabs/noise2noise

Train once, test in varying imaging configurations (types) & noise levels.
Dataset: 
    training set: mixed noise levels, microscopies and cells
    test set: mixed
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from models.unet import UnetN2N, UnetN2Nv2
from utils.metrics import cal_psnr
from utils.data_loader import (load_denoising_n2n_train, 
                               load_denoising_test_mix, fluore_to_tensor)
from utils.practices import OneCycleScheduler, adjust_learning_rate, find_lr
from utils.misc import mkdirs, module_size
from utils.plot import save_samples, save_stats
import numpy as np
import argparse
import json
import random
import time
import sys
from pprint import pprint
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Training N2N')
        self.add_argument('--exp-name', type=str, default='n2n', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')        
        self.add_argument('--post', action='store_true', default=False, help='post proc mode')
        self.add_argument('--debug', action='store_true', default=False, help='verbose stdout')
        self.add_argument('--net', type=str, default='unet', choices=['unet', 'unetv2'])
        # data
        self.add_argument('--data-root', type=str, default="./dataset", help='directory to dataset root')
        self.add_argument('--imsize', type=int, default=256)
        self.add_argument('--in-channels', type=int, default=1)
        self.add_argument('--out-channels', type=int, default=1)
        self.add_argument('--transform', type=str, default='four_crop', choices=['four_crop', 'center_crop'])
        self.add_argument('--noise-levels-train', type=list, default=[1, 2, 4, 8, 16])
        self.add_argument('--noise-levels-test', type=list, default=[1])
        self.add_argument('--test-group', type=int, default=19)
        self.add_argument('--captures', type=int, default=50, help='how many captures in each group to load')
        # training
        self.add_argument('--epochs', type=int, default=400, help='number of iterations to train')
        self.add_argument('--batch-size', type=int, default=4, help='input batch size for training')
        self.add_argument('--lr', type=float, default=1e-4, help='learnign rate')
        self.add_argument('--wd', type=float, default=0., help="weight decay")
        self.add_argument('--test-batch-size', type=int, default=2, help='input batch size for testing')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        self.add_argument('--cuda', type=int, default=0, help='cuda number')
        # logging
        self.add_argument('--ckpt-freq', type=int, default=50, help='how many epochs to wait before saving model')
        self.add_argument('--print-freq', type=int, default=100, help='how many minibatches to wait before printing training status')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-epochs', type=int, default=5, help='how many epochs to wait before plotting test output')
        self.add_argument('--cmap', type=str, default='inferno', help='attach notes to the run dir')

    def parse(self):
        args = self.parse_args()
        date = '{}'.format(time.strftime('%b_%d'))
        args.run_dir = args.exp_dir + '/' + args.exp_name + '/' + date \
            + f'/{args.net}_noise_train{args.noise_levels_train}_'\
            f'test{args.noise_levels_test}_{args.transform}_'\
            f'epochs{args.epochs}_bs{args.batch_size}_lr{args.lr}'
        args.ckpt_dir = args.run_dir + '/checkpoints'

        if not args.post:
            mkdirs([args.run_dir, args.ckpt_dir])

        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark=True

        print('Arguments:')
        pprint(vars(args))

        if not args.post:
            with open(args.run_dir + "/args.txt", 'w') as args_file:
                json.dump(vars(args), args_file, indent=4)

        return args

args = Parser().parse()
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])
if args.net == 'unet':
    model = UnetN2N(args.in_channels, args.out_channels).to(device)
elif args.net == 'unetv2':
    model = UnetN2Nv2(args.in_channels, args.out_channels).to(device)

if args.debug:
    print(model)
    print(model.model_size)

if args.transform == 'four_crop':
    # wide field images may have complete noise in center-crop case
    transform = transforms.Compose([
        transforms.FiveCrop(args.imsize),
        transforms.Lambda(lambda crops: torch.stack([
            fluore_to_tensor(crop) for crop in crops[:4]])),
        transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
        ])
elif args.transform == 'center_crop':
    # default transform
    transform = None

train_loader = load_denoising_n2n_train(args.data_root,
    batch_size=args.batch_size, noise_levels=args.noise_levels_train, 
    types=None, transform=transform, target_transform=transform, 
    patch_size=args.imsize, test_fov=args.test_group)

test_loader = load_denoising_test_mix(args.data_root, 
    batch_size=args.test_batch_size, noise_levels=args.noise_levels_test, 
    transform=transform, patch_size=args.imsize)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
    weight_decay=args.wd, betas=[0.9, 0.99])
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
scheduler = OneCycleScheduler(lr_max=args.lr, div_factor=10, pct_start=0.3)

multiplier = 4 if args.transform == 'four_crop' else 1
n_train_samples = len(train_loader.dataset) * multiplier
n_test_samples = len(test_loader.dataset) * multiplier
pixels_per_sample = train_loader.dataset[0][0].numel()
n_train_pixels = n_train_samples * pixels_per_sample
n_test_pixels = n_test_samples * pixels_per_sample

np.random.seed(113)
fixed_idx = np.random.permutation(len(test_loader.dataset))[:8]
print(f'fixed test index: {fixed_idx}')

fixed_test_noisy = torch.stack([(test_loader.dataset[i][0]) for i in fixed_idx])
fixed_test_clean = torch.stack([(test_loader.dataset[i][1]) for i in fixed_idx])
if args.transform == 'four_crop':
    fixed_test_noisy = fixed_test_noisy[:, -1]
    fixed_test_clean = fixed_test_clean[:, -1]
print(f'fixed test noisy shape: {fixed_test_noisy.shape}')
fixed_test_noisy = fixed_test_noisy.to(device)

logger = {}
logger['psnr_train'] = []
logger['rmse_train'] = []
logger['psnr_test'] = []
logger['rmse_test'] = []

total_steps = args.epochs * len(train_loader)
print('Start training........................................................')
try:
    tic = time.time()
    iters = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        # if epoch == 1:
        #     print('start finding lr...')
        #     log_lrs, losses = find_lr(model, train_loader, optimizer, 
        #         F.mse_loss, device=device)
        #     plt.plot(log_lrs[10:-5],losses[10:-5])
        #     plt.savefig('find_lr_n2n.png')
        #     plt.close()
        #     sys.exit(0)
        psnr, mse = 0., 0.
        for batch_idx, (noisy_input, noisy_target, clean) in enumerate(train_loader):
            iters += 1
            noisy_input, noisy_target, clean = noisy_input.to(device), \
                noisy_target.to(device), clean.to(device)
            
            if args.transform == 'four_crop':
                # fuse batch and four crop
                noisy_input = noisy_input.view(-1, *noisy_input.shape[2:])
                noisy_target = noisy_target.view(-1, *noisy_target.shape[2:])
                clean = clean.view(-1, *clean.shape[2:])
            
            model.zero_grad()
            denoised = model(noisy_input)
            loss = F.mse_loss(denoised, noisy_target, reduction='sum')
            loss.backward()

            step = epoch * len(train_loader) + batch_idx + 1
            pct = step / total_steps
            lr = scheduler.step(pct)
            adjust_learning_rate(optimizer, lr)

            optimizer.step()

            mse += loss.item()
            with torch.no_grad():
                psnr += cal_psnr(clean, denoised.detach()).sum().item()
            if iters % args.print_freq == 0:
                print(f'[{batch_idx+1}|{len(train_loader)}]'\
                    f'[{epoch}|{args.epochs}] training PSNR: '\
                    f'{(psnr / (batch_idx+1) / args.batch_size / multiplier):.6f}')
        print(f'Epoch {epoch}, lr {lr}')
         
        psnr = psnr / n_train_samples
        rmse = np.sqrt(mse / n_train_pixels)
        scheduler.step(psnr)
        
        if epoch % args.log_freq == 0:
            logger['psnr_train'].append(psnr)
            logger['rmse_train'].append(rmse)
        print("Epoch {} training PSNR: {:.6f}, RMSE: {:.6f}".format(epoch, psnr, rmse))

        # save model
        if epoch % args.ckpt_freq == 0:
            torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))

        # test ------------------------------
        with torch.no_grad():
            model.eval()
            psnr, mse = 0., 0.
            for batch_idx, (noisy, clean) in enumerate(test_loader):
                noisy, clean = noisy.to(device), clean.to(device)
                if args.transform == 'four_crop':
                    # fuse batch and four crop
                    noisy = noisy.view(-1, *noisy.shape[2:])
                    clean = clean.view(-1, *clean.shape[2:])
                denoised = model(noisy)
                loss = F.mse_loss(denoised, clean, reduction='sum')
                mse += loss.item()
                psnr += cal_psnr(clean, denoised).sum().item()

            psnr = psnr / n_test_samples
            rmse = np.sqrt(mse / n_test_pixels)
            
            if epoch % args.plot_epochs == 0:
                print('Epoch {}: plot test denoising [input, denoised, clean, denoised - clean]'.format(epoch))
                samples = torch.cat((noisy[:4], denoised[:4], clean[:4], denoised[:4] - clean[:4]))
                save_samples(args.pred_dir, samples, epoch, 'test', epoch=True, cmap=args.cmap)
                # fixed test
                fixed_denoised = model(fixed_test_noisy)
                samples = torch.cat((fixed_test_noisy[:4].cpu(), 
                    fixed_denoised[:4].cpu(), fixed_test_clean[:4], 
                    fixed_denoised[:4].cpu() - fixed_test_clean[:4]))
                save_samples(args.pred_dir, samples, epoch, 'fixed_test1', epoch=True, cmap=args.cmap)
                samples = torch.cat((fixed_test_noisy[4:8].cpu(), 
                    fixed_denoised[4:8].cpu(), fixed_test_clean[4:8],
                    fixed_denoised[4:8].cpu() - fixed_test_clean[4:8]))
                save_samples(args.pred_dir, samples, epoch, 'fixed_test2', epoch=True, cmap=args.cmap)

            if epoch % args.log_freq == 0:
                logger['psnr_test'].append(psnr)
                logger['rmse_test'].append(rmse)
            print("Epoch {}: test PSNR: {:.6f}, RMSE: {:.6f}".format(epoch, psnr, rmse))
                
    tic2 = time.time()
    print("Finished training {} epochs using {} seconds"
        .format(args.epochs, tic2 - tic))

    x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
    # plot the rmse, r2-score curve and save them in txt
    save_stats(args.train_dir, logger, x_axis, 'psnr_train', 'psnr_test', 
        'rmse_train', 'rmse_test')

    args.training_time = tic2 - tic
    args.n_params, args.n_layers = module_size(model)
    with open(args.run_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)

except KeyboardInterrupt:
    print('Keyboard Interrupt captured...Saving models & training logs')
    tic2 = time.time()
    torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))
    x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
    # plot the rmse, r2-score curve and save them in txt
    save_stats(args.train_dir, logger, x_axis, 'psnr_train', 'psnr_test', 
        'rmse_train', 'rmse_test')

    args.training_time = tic2 - tic
    args.n_params, args.n_layers = module_size(model)
    with open(args.run_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)
