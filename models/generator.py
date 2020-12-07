from models.blocks import GLU, InputBlock, OutputBlock, UpSamplingBlock, SkipLayerExcitationBlock

class Generator(nn.Module):
    def __init__(self, out_resolution):
        super(Generator, self).__init__()
        
        self.resolution = out_resolution
        self.resolutionswise = {256:32, 512:16, 1024:8} 

        self.input_block = InputBlock(256, 1024) 

        self.upsample_8 = UpSamplingBlock(1024, 512)
        self.upsample_16 = UpSamplingBlock(512, 256)
        self.upsample_32 = UpSamplingBlock(256, 256)
        self.upsample_64 = UpSamplingBlock(256, 128)
        self.upsample_128 = UpSamplingBlock(128, 64)
        self.upsample_256 = UpSamplingBlock(64, 32)
        self.upsample_512 = UpSamplingBlock(32, 16)
        self.upsample_1024 = UpSamplingBlock(16, 8)

        self.sle_8_128 = SkipLayerExcitationBlock(512,64)
        self.sle_16_256 = SkipLayerExcitationBlock(256,32)
        self.sle_32_512 = SkipLayerExcitationBlock(256,16)

        self.out = OutputBlock(self.resolutionswise[self.resolution])

    def forward(self, x):
        x = self.input_block(x) # --> (B, 1024, 4, 4)

        x_8 = self.upsample_8(x)  # --> (B, 512, 8, 8)
        x_16 = self.upsample_16(x_8)  # --> (B, 256, 16, 16)
        x_32 = self.upsample_32(x_16)  # --> (B, 128, 32, 32)
        x_64 = self.upsample_64(x_32)  # --> (B, 128, 64, 64)

        x_128 = self.upsample_128(x_64)  # --> (B, 64, 128, 128)
        x_sle_128 = self.sle_8_128([x_8, x_128]) # --> (B, 64, 128, 128)

        x_256 = self.upsample_256(x_sle_128)  # --> (B, 32, 256, 256)
        x = self.sle_16_256([x_16, x_256]) # --> (B, 32, 256, 256)

        if self.resolution >= 512:
            x_512 = self.upsample_512(x)  # --> (B, 16, 512, 512)
            x = self.sle_32_512([x_32, x_512])  # --> (B, 16, 512, 512) 

            if self.resolution == 1024:
                x = self.upsample_1024(x) # --> (B, 8, 1024, 1024)

        x = self.out(x)  # --> (B, 3, res, res)
        return x
