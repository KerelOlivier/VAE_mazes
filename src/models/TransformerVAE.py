"""
Transformer VAE model.

Combination of ResNet (down/up) blocks and Attention blocks.
"""
from src.models.VAE import VAE, IEncoder, IDecoder, IPrior
from torch import nn

#################
# Encoder Block #
#################

class TransformerEncoder(IEncoder):
    def __init__(self):
        super().__init__()

class DownSample2D(nn.Modeule):
    def __init__(self):
        super().__init__()

#################
# Decoder block #
#################

class TransformerDecoder(IDecoder):
    def __init__(self):
        super().__init__()


class Upsample2D(nn.Module):
    def __init__(self):
        super().__init__()


##################
# UNet mid block #
##################

class TransformerMid(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        num_blocks):
        """
        in_channels: number of input channels
        out_channels: number of output channels
        num_blocks: number of residual, attention block pairs
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        # First residual block to scale the data
        self.residual = ResidualBlock(self.in_channels, self.out_channels)

        # remaining blocks that don't scale
        self.blocks = [(AttentionBlock(), ResidualBlock(self.out_channels, self.out_channels)) for _ in range(self.num_blocks))]

    def forward(x):
        x = self.residual(x)

        for attn, res in blocks:
            x = attn(x)
            x = res(x)
            
        return x


###############
# Base Blocks #
###############

class ResidualBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels
        kernel_size = 3):
        """
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size of the embedding convolutional layers
        """
        super().__init__():
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, padding=1)
        self.conv_con = nn.Conv2d(self.in_channels, self.out_channels, 1)


    def forward(self, x):
        assert x.shape[0] == self.in_channels
        z = self.conv_con(x) #residual

        y = self.conv1(x)
        y = y.relu()
        y = self.conv2(y)
        y += z
        return y.relu()


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
