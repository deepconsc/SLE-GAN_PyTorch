import torch.nn as nn 
import torch
import torch.nn.functional as F
from modules.disc_blocks import InputBlock, DownSamplingBlock, Decoder, RealFakeOutputBlock
from random import randint

class Discriminator(nn.Module):
    def __init__(self, resolution):
        super(Discriminator, self).__init__()
        assert resolution in [256, 512, 1024]

        
        downsampling = {256: 1, 512: 2, 1024: 4}
        out_channels = {256: 8, 512: 16, 1024: 32}
        
        self.input_block = InputBlock(downsampling_factor=downsampling[resolution], out_channels=out_channels[resolution])

        self.downsample_128 = DownSamplingBlock(in_channels=out_channels[resolution], out_channels=64)
        self.downsample_64 = DownSamplingBlock(in_channels=64, out_channels=128)
        self.downsample_32 = DownSamplingBlock(in_channels=128, out_channels=128)
        self.downsample_16 = DownSamplingBlock(in_channels=128, out_channels=256)
        self.downsample_8 = DownSamplingBlock(in_channels=256, out_channels=512)

        self.decoder_image_part = Decoder(in_channels=256, out_channels=3)
        self.decoder_image = Decoder(in_channels=512, out_channels=3)

        self.logits = RealFakeOutputBlock(in_channels=512, out_channels=1)

 
    def forward(self, x, random):
        x = self.input_block(x)  # --> (B, 8/16/32, 256, 256)
        x = self.downsample_128(x)  # --> (B, 64, 128, 128)
        x = self.downsample_64(x)  # --> (B, 128, 64, 64)
        x = self.downsample_32(x)  # --> (B, 128, 32, 32)
        x_16 = self.downsample_16(x)  # --> (B, 256, 16, 16)
        x_8 = self.downsample_8(x_16)  # --> (B, 512, 8, 8)

        x_image_decoded_128_x_16 = self.decoder_image_part(x_16[:,:,random-4:random+4,random-4:random+4])  # --> (B, 3, 128, 128)
        x_image_decoded_128_x_8 = self.decoder_image(x_8)  # --> (B, 3, 128, 128)

        logits = self.logits(x_8)  # --> (B, 1, 5, 5)

        return logits, x_image_decoded_128_x_8, x_image_decoded_128_x_16
