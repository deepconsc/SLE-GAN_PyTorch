import torch.nn as nn 
import torch
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channels = x.shape[1] // 2
        x_1 = x[:, :channels, :, :]
        x_2 = x[:, channels:, :, :]
        return x_1 * self.sigmoid(x_2)


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()

        """
        When activation is GLU, we should double the output channels 
        to match the desired one, as long as ch -> GLU -> ch/2
        """

        self.transpose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=(4,4), stride=1)
        self.batchnorm = nn.BatchNorm2d(out_channels*2)
        self.glu = GLU()

    def forward(self, x):
        x = self.transpose(x)
        x = self.batchnorm(x)
        x = self.glu(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self, in_channels):
        super(OutputBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=(1,1))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSamplingBlock, self).__init__()

        """
        When activation is GLU, we should double the output channels 
        to match the desired one, as long as ch -> GLU -> ch/2
        """

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2) #nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=(1,1))
        self.batchnorm = nn.BatchNorm2d(out_channels*2)
        self.glu = GLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.glu(x)
        return x 

class SkipLayerExcitationBlock(nn.Module):
    def __init__(self, low_in, low_out):
        super(SkipLayerExcitationBlock, self).__init__()

        self.leaky = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.conv2d_low = nn.Conv2d(in_channels=low_in, out_channels=low_out, kernel_size=(4,4))
        self.conv2d_high = nn.Conv2d(in_channels=low_out, out_channels=low_out, kernel_size=(1,1))
        self.pool = nn.AdaptiveAvgPool2d(4)


    def forward(self, x):
        x_low, x_high = x

        x = self.pool(x_low)
        x = self.conv2d_low(x)
        x = self.leaky(x)
        x = self.conv2d_high(x)
        x = self.sigmoid(x)
        return x * x_high

  

