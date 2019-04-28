import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
import numpy as np
from .misc import to_numpy
import torchvision.utils
plt.switch_backend('agg')


pub = False
if pub:
    ext = 'pdf'
    dpi = 300
else:
    ext = 'png'
    dpi = None


def save_samples(save_dir, images, iters, name, nrow=4, heatmap=True, 
    cmap='gray', epoch=False):
    """Save samples in grid as images or plots
    Args:
        images (Tensor): B x C x H x W
    """
    images = to_numpy(images)

    step = 'epoch' if epoch else 'iter'

    if images.shape[0] < 10:
        nrow = 2
        ncol = images.shape[0] // nrow
    else:
        ncol = nrow

    if heatmap:
        for c in range(images.shape[1]):
            fig = plt.figure(1, (11, 12))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(nrow, ncol),
                             axes_pad=0.1,
                             share_all=False,
                             cbar_location="top",
                             cbar_mode="single",
                             cbar_size="3%",
                             cbar_pad=0.1
                             )
            for j, ax in enumerate(grid):
                im = ax.imshow(images[j][c], cmap=cmap, origin='upper')
                ax.set_axis_off()
                ax.set_aspect('equal')
            cbar = grid.cbar_axes[0].colorbar(im)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.toggle_label(True)
            plt.savefig(save_dir + '/{}_c{}_{}{}.png'.format(name, c, step, iters),
                        bbox_inches='tight')
            plt.close(fig)
    else:
        torchvision.utils.save_image(images, 
                          save_dir + '/fake_samples_{}{}.png'.format(step, iters),
                          nrow=nrow,
                          normalize=True)


def save_stats(save_dir, logger, x_axis, *metrics):
    """
    Args:
        metrics (list of strings): e.g. ['loss_d', 'loss_g', 'rmse_test', 'mae_train']
    """
    # just in case
    for metric in metrics:
        metric_arr = logger[metric]
        np.savetxt(save_dir + '/{}.txt'.format(metric), metric_arr)


    if set(['rmse_train', 'rmse_test']) <  set(metrics):
        rmse_train = logger['rmse_train']
        rmse_test = logger['rmse_test']
        
        plt.figure()
        plt.plot(x_axis, rmse_train, label="train: {:.3f}".format(np.mean(rmse_train[-5:])))
        plt.plot(x_axis, rmse_test, label="test: {:.3f}".format(np.mean(rmse_test[-5:])))
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend(loc='upper right')
        plt.savefig(save_dir + "/rmse.pdf", dpi=300)
        plt.close()

    if set(['psnr_train', 'psnr_test']) <  set(metrics):
        psnr_train = logger['psnr_train']
        psnr_test = logger['psnr_test']
        
        plt.figure()
        plt.plot(x_axis, psnr_train, label="train: {:.3f}".format(np.mean(psnr_train[-5:])))
        plt.plot(x_axis, psnr_test, label="test: {:.3f}".format(np.mean(psnr_test[-5:])))
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.legend(loc='upper right')
        plt.savefig(save_dir + "/psnr.pdf", dpi=300)
        plt.close()


def plot_row(arrs, save_dir, filename, same_range=False, plot_fn='imshow', 
    cmap='viridis', colorbar=True):
    """
    Args:
        arrs (sequence of 2D Tensor or Numpy): seq of arrs to be plotted
        save_dir (str):
        filename (str):
        same_range (bool): if True, subplots have the same range (colorbar)
        plot_fn (str): choices=['imshow', 'contourf']
        colorbar (bool): add colorbar or not
    """
    interpolation = None
    arrs = [to_numpy(arr) for arr in arrs]

    if same_range:
        vmax = max([np.amax(arr) for arr in arrs])
        vmin = min([np.amin(arr) for arr in arrs])
    else:
        vmax, vmin = None, None

    fig, _ = plt.subplots(1, len(arrs), figsize=(4.4 * len(arrs), 4))
    for i, ax in enumerate(fig.axes):
        if plot_fn == 'imshow':
            cax = ax.imshow(arrs[i], cmap=cmap, interpolation=interpolation,
                            vmin=vmin, vmax=vmax)
        elif plot_fn == 'contourf':
            cax = ax.contourf(arrs[i], 50, cmap=cmap, vmin=vmin, vmax=vmax)
        if plot_fn == 'contourf':
            for c in cax.collections:
                c.set_edgecolor("face")
                c.set_linewidth(0.000000000001)
        ax.set_axis_off()
        if colorbar:
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.ax.yaxis.set_offset_position('left')
            # cbar.ax.tick_params(labelsize=5)
            cbar.update_ticks()
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.savefig(save_dir + f'/{filename}.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
