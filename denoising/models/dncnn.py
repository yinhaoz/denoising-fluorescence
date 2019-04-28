import torch
import torch.nn as nn
import torch.nn.init as init


class DnCNN(nn.Module):
    # reference: https://github.com/cszn/DnCNN/tree/master/TrainingCodes/dncnn_pytorch
    def __init__(self, depth=17, n_channels=64, image_channels=1, 
        use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(image_channels, n_channels, 
            kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(n_channels, n_channels, 
            kernel_size=kernel_size, padding=padding, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, 
            kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

        print(f'model size: {self._model_size()}')

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        # out: residual, y: noisy input
        return y - out

    def _initialize_weights(self):
        print('init weight')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _model_size(self):
        n_params, n_conv_layers = 0, 0
        for param in self.parameters():
            n_params += param.numel()
        for module in self.modules():
            if 'Conv' in module.__class__.__name__ \
                    or 'conv' in module.__class__.__name__:
                n_conv_layers += 1
        return n_params, n_conv_layers



class DnCNN_NRL(nn.Module):
    """Non-residual learning.
    """
    def __init__(self, depth=17, n_channels=64, image_channels=1, 
        use_bnorm=True, kernel_size=3):
        super(DnCNN_NRL, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(image_channels, n_channels, 
            kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(n_channels, n_channels, 
            kernel_size=kernel_size, padding=padding, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, 
            kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

        print(f'model size: {self._model_size()}')

    def forward(self, x):
        # output denoised directly
        return self.dncnn(x)

    def _initialize_weights(self):
        print('init weight')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _model_size(self):
        n_params, n_conv_layers = 0, 0
        for param in self.parameters():
            n_params += param.numel()
        for module in self.modules():
            if 'Conv' in module.__class__.__name__ \
                    or 'conv' in module.__class__.__name__:
                n_conv_layers += 1
        return n_params, n_conv_layers