import torch.nn as nn 
import torch
import torch.nn.functional as F

class InputBlock(nn.Module):
    def __init__(self, downsampling_factor, out_channels):
        super(InputBlock, self).__init__()

        assert downsampling_factor in [1, 2, 4]

        conv1_stride = 2
        conv2_stride = 2

        if downsampling_factor <= 2:
            conv2_stride = 1

        if downsampling_factor == 1:
            conv1_stride = 1

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=(1,1), stride=conv1_stride)
        self.leaky = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1), stride=conv2_stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.leaky(x)
        return x



class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        
        # Block 1
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=2)
        self.conv1_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1), stride=1)

        # Block 2 

        self.pool = nn.AvgPool2d(2)
        self.conv2_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1))

    def forward(self, x):
        y = x

        x = self.conv1_1(x)
        x = self.batchnorm(x)
        x = self.leaky(x)
        x = self.conv1_2(x)
        x = self.batchnorm(x)
        x = self.leaky(x)

        y = self.pool(y)
        y = self.conv2_1(y)
        y = self.batchnorm(y)
        y = self.leaky(y)
        return x + y 





class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.ks = 1
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.glu = GLU()
        self.tanh = nn.Tanh()

        self.conv_256 = nn.Conv2d(in_channels=in_channels, out_channels=256*2, kernel_size=self.ks) 
        self.batchnorm_256 = nn.BatchNorm2d(256*2)

        self.conv_128_1 = nn.Conv2d(in_channels=256, out_channels=128*2, kernel_size=1) 
        self.batchnorm_128_1 = nn.BatchNorm2d(128*2)

        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128*2, kernel_size=self.ks) 
        self.batchnorm_128_2 = nn.BatchNorm2d(128*2)

        self.conv_64 = nn.Conv2d(in_channels=128, out_channels=64*2, kernel_size=self.ks) 
        self.batchnorm_64 = nn.BatchNorm2d(64*2)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(1,1))  

    def forward(self, x):
        x = self.upsample(x)

        x = self.conv_256(x)
        x = self.batchnorm_256(x)
        x = self.glu(x)

        x = self.upsample(x)
        x = self.conv_128_1(x)
        x = self.batchnorm_128_1(x)
        x = self.glu(x)

        x = self.upsample(x)
        x = self.conv_128_2(x)
        x = self.batchnorm_128_2(x)
        x = self.glu(x)

        x = self.upsample(x)
        x = self.conv_64(x)
        x = self.batchnorm_64(x)
        x = self.glu(x)

        x = self.conv_output(x)
        x = self.tanh(x)
        return x




class RealFakeOutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RealFakeOutputBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) 
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=4) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.leaky(x)
        x = self.conv2(x)
        return x

