import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


class UpsamplingNearest2d(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class UnetN2N(nn.Module):
    """https://github.com/NVlabs/noise2noise
    Lehtinen, Jaakko, et al. "Noise2Noise: Learning Image Restoration without 
    Clean Data." arXiv preprint arXiv:1803.04189 (2018).
    """
    def __init__(self, in_channels, out_channels):
        super(UnetN2N, self).__init__()

        self.enc_conv0 = conv3x3(in_channels, 48)
        self.enc_relu0 = nn.LeakyReLU(0.1)
        self.enc_conv1 = conv3x3(48, 48)
        self.enc_relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 128
        self.enc_conv2 = conv3x3(48, 48)
        self.enc_relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 64
        self.enc_conv3 = conv3x3(48, 48)
        self.enc_relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 32
        self.enc_conv4 = conv3x3(48, 48)
        self.enc_relu4 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # 16
        self.enc_conv5 = conv3x3(48, 48)
        self.enc_relu5 = nn.LeakyReLU(0.1)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        # 8
        self.enc_conv6 = conv3x3(48, 48)
        self.enc_relu6 = nn.LeakyReLU(0.1)
        self.upsample5 = UpsamplingNearest2d(scale_factor=2)
        # 16
        self.dec_conv5a = conv3x3(96, 96)
        self.dec_relu5a = nn.LeakyReLU(0.1)
        self.dec_conv5b = conv3x3(96, 96)
        self.dec_relu5b = nn.LeakyReLU(0.1)
        self.upsample4 = UpsamplingNearest2d(scale_factor=2)
        # 32
        self.dec_conv4a = conv3x3(144, 96)
        self.dec_relu4a = nn.LeakyReLU(0.1)
        self.dec_conv4b = conv3x3(96, 96)
        self.dec_relu4b = nn.LeakyReLU(0.1)
        self.upsample3 = UpsamplingNearest2d(scale_factor=2)
        # 64
        self.dec_conv3a = conv3x3(144, 96)
        self.dec_relu3a = nn.LeakyReLU(0.1)
        self.dec_conv3b = conv3x3(96, 96)
        self.dec_relu3b = nn.LeakyReLU(0.1)
        self.upsample2 = UpsamplingNearest2d(scale_factor=2)
        # 128
        self.dec_conv2a = conv3x3(144, 96)
        self.dec_relu2a = nn.LeakyReLU(0.1)
        self.dec_conv2b = conv3x3(96, 96)
        self.dec_relu2b = nn.LeakyReLU(0.1)
        self.upsample1 = UpsamplingNearest2d(scale_factor=2)
        # 256
        self.dec_conv1a = conv3x3(96 + in_channels, 64)
        self.dec_relu1a = nn.LeakyReLU(0.1)
        self.dec_conv1b = conv3x3(64, 32)
        self.dec_relu1b = nn.LeakyReLU(0.1)
        self.dec_conv1c = conv3x3(32, out_channels)

    def forward(self, x):
        out_pool1 = self.pool1(self.enc_relu1(self.enc_conv1(self.enc_relu0(self.enc_conv0(x)))))
        out_pool2 = self.pool2(self.enc_relu2(self.enc_conv2(out_pool1)))
        out_pool3 = self.pool3(self.enc_relu3(self.enc_conv3(out_pool2)))
        out_pool4 = self.pool4(self.enc_relu4(self.enc_conv4(out_pool3)))
        out_pool5 = self.pool5(self.enc_relu5(self.enc_conv5(out_pool4)))
        out = self.upsample5(self.enc_relu6(self.enc_conv6(out_pool5)))
        out = self.upsample4(self.dec_relu5b(self.dec_conv5b(self.dec_relu5a(self.dec_conv5a(torch.cat((out, out_pool4), 1))))))
        out = self.upsample3(self.dec_relu4b(self.dec_conv4b(self.dec_relu4a(self.dec_conv4a(torch.cat((out, out_pool3), 1))))))
        out = self.upsample2(self.dec_relu3b(self.dec_conv3b(self.dec_relu3a(self.dec_conv3a(torch.cat((out, out_pool2), 1))))))
        out = self.upsample1(self.dec_relu2b(self.dec_conv2b(self.dec_relu2a(self.dec_conv2a(torch.cat((out, out_pool1), 1))))))
        out = self.dec_conv1c(self.dec_relu1b(self.dec_conv1b(self.dec_relu1a(self.dec_conv1a(torch.cat((out, x), 1))))))
        return out



class UnetN2Nv2(nn.Module):
    """Add BatchNorm and Tanh
    Lehtinen, Jaakko, et al. "Noise2Noise: Learning Image Restoration without 
    Clean Data." arXiv preprint arXiv:1803.04189 (2018).
    Add BatchNorm and Tanh out activation
    """
    def __init__(self, in_channels, out_channels):
        super(UnetN2Nv2, self).__init__()

        self.enc_conv0 = conv3x3(in_channels, 48)
        self.enc_relu0 = nn.LeakyReLU(0.1)
        self.enc_conv1 = conv3x3(48, 48)
        self.enc_bn1 = nn.BatchNorm2d(48)
        self.enc_relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 128
        self.enc_conv2 = conv3x3(48, 48)
        self.enc_bn2 = nn.BatchNorm2d(48)
        self.enc_relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 64
        self.enc_conv3 = conv3x3(48, 48)
        self.enc_bn3 = nn.BatchNorm2d(48)
        self.enc_relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 32
        self.enc_conv4 = conv3x3(48, 48)
        self.enc_bn4 = nn.BatchNorm2d(48)
        self.enc_relu4 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        # 16
        self.enc_conv5 = conv3x3(48, 48)
        self.enc_bn5 = nn.BatchNorm2d(48)
        self.enc_relu5 = nn.LeakyReLU(0.1)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        # 8
        self.enc_conv6 = conv3x3(48, 48)
        self.enc_bn6 = nn.BatchNorm2d(48)
        self.enc_relu6 = nn.LeakyReLU(0.1)
        self.upsample5 = UpsamplingNearest2d(scale_factor=2)
        # 16
        self.dec_conv5a = conv3x3(96, 96)
        self.dec_bn5a = nn.BatchNorm2d(96)
        self.dec_relu5a = nn.LeakyReLU(0.1)
        self.dec_conv5b = conv3x3(96, 96)
        self.dec_bn5b = nn.BatchNorm2d(96)
        self.dec_relu5b = nn.LeakyReLU(0.1)
        self.upsample4 = UpsamplingNearest2d(scale_factor=2)
        # 32
        self.dec_conv4a = conv3x3(144, 96)
        self.dec_bn4a = nn.BatchNorm2d(96)
        self.dec_relu4a = nn.LeakyReLU(0.1)
        self.dec_conv4b = conv3x3(96, 96)
        self.dec_bn4b = nn.BatchNorm2d(96)
        self.dec_relu4b = nn.LeakyReLU(0.1)
        self.upsample3 = UpsamplingNearest2d(scale_factor=2)
        # 64
        self.dec_conv3a = conv3x3(144, 96)
        self.dec_bn3a = nn.BatchNorm2d(96)
        self.dec_relu3a = nn.LeakyReLU(0.1)
        self.dec_conv3b = conv3x3(96, 96)
        self.dec_bn3b = nn.BatchNorm2d(96)
        self.dec_relu3b = nn.LeakyReLU(0.1)
        self.upsample2 = UpsamplingNearest2d(scale_factor=2)
        # 128
        self.dec_conv2a = conv3x3(144, 96)
        self.dec_bn2a = nn.BatchNorm2d(96)
        self.dec_relu2a = nn.LeakyReLU(0.1)
        self.dec_conv2b = conv3x3(96, 96)
        self.dec_bn2b = nn.BatchNorm2d(96)
        self.dec_relu2b = nn.LeakyReLU(0.1)
        self.upsample1 = UpsamplingNearest2d(scale_factor=2)
        # 256
        self.dec_conv1a = conv3x3(96 + in_channels, 64)
        self.dec_bn1a = nn.BatchNorm2d(64)
        self.dec_relu1a = nn.LeakyReLU(0.1)
        self.dec_conv1b = conv3x3(64, 32)
        self.dec_bn1b = nn.BatchNorm2d(32)
        self.dec_relu1b = nn.LeakyReLU(0.1)
        self.dec_conv1c = conv3x3(32, out_channels)
        self.dec_act = nn.Tanh()

    def forward(self, x):
        out_pool1 = self.pool1(self.enc_relu1(self.enc_bn1(self.enc_conv1(self.enc_relu0(self.enc_conv0(x))))))
        out_pool2 = self.pool2(self.enc_relu2(self.enc_bn2(self.enc_conv2(out_pool1))))
        out_pool3 = self.pool3(self.enc_relu3(self.enc_bn3(self.enc_conv3(out_pool2))))
        out_pool4 = self.pool4(self.enc_relu4(self.enc_bn4(self.enc_conv4(out_pool3))))
        out_pool5 = self.pool5(self.enc_relu5(self.enc_bn5(self.enc_conv5(out_pool4))))
        out = self.upsample5(self.enc_relu6(self.enc_bn6(self.enc_conv6(out_pool5))))
        out = self.upsample4(self.dec_relu5b(self.dec_bn5b(self.dec_conv5b(self.dec_relu5a(self.dec_bn5a(self.dec_conv5a(torch.cat((out, out_pool4), 1))))))))
        out = self.upsample3(self.dec_relu4b(self.dec_bn4b(self.dec_conv4b(self.dec_relu4a(self.dec_bn4a(self.dec_conv4a(torch.cat((out, out_pool3), 1))))))))
        out = self.upsample2(self.dec_relu3b(self.dec_bn3b(self.dec_conv3b(self.dec_relu3a(self.dec_bn3a(self.dec_conv3a(torch.cat((out, out_pool2), 1))))))))
        out = self.upsample1(self.dec_relu2b(self.dec_bn2b(self.dec_conv2b(self.dec_relu2a(self.dec_bn2a(self.dec_conv2a(torch.cat((out, out_pool1), 1))))))))
        out = self.dec_conv1c(self.dec_relu1b(self.dec_bn1b(self.dec_conv1b(self.dec_relu1a(self.dec_bn1a(self.dec_conv1a(torch.cat((out, x), 1))))))))
        out = self.dec_act(out)
        return out


    @property
    def model_size(self):
        return self._model_size()

    def _model_size(self):
        n_params, n_conv_layers = 0, 0
        for param in self.parameters():
            n_params += param.numel()
        for module in self.modules():
            if 'Conv' in module.__class__.__name__ \
                    or 'conv' in module.__class__.__name__:
                n_conv_layers += 1
        return n_params, n_conv_layers



if __name__ == '__main__':

    unet = UnetN2N(3, 3).to('cuda:2')
    print(unet)
    print(unet.model_size)

    x = torch.randn(16, 3, 256, 256).to('cuda:2')
    y = unet(x)
    print(y.shape)
