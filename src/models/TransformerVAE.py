"""
Transformer VAE model.

Combination of ResNet (down/up) blocks and Attention blocks.
"""
import torch

from src.models.VAE import VAE, IEncoder, IDecoder, IPrior
from torch import nn
from torch.nn.functional import interpolate
import numpy as np


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


class UpSample2D(nn.Module):
    # using solutions to checkerboard artifacts discussed here: https://distill.pub/2016/deconv-checkerboard/
    def __init__(self, channels, up_sample_factor=2.0, kernel_size=3):
        """
        Upsample 2D convolution
        :param up_sample_factor: Factor by which to up sample the input image
        """
        super().__init__()
        self.up_sample_factor = up_sample_factor

        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = interpolate(x, scale_factor=self.up_sample_factor, mode='bilinear')
        x = self.conv(x)
        return x


##################
# UNet mid block #
##################

class TransformerMid(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks,
                 num_heads,
                 rescale_output_factor: float = 1.0,
                 kernel_size=3):
        """
        in_channels: number of input channels
        out_channels: number of output channels
        num_blocks: number of residual, attention block pairs
        num_heads: number of attention heads for the attention layers
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.rescale_output_factor = rescale_output_factor
        self.kernel_size = kernel_size

        # First residual block to scale the data
        self.residual = ResidualBlock(self.in_channels, self.out_channels)

        # remaining blocks that don't scale
        self.blocks = [
            (AttentionBlock(self.out_channels, self.num_heads, rescale_output_factor=self.rescale_output_factor),
             ResidualBlock(self.out_channels, self.out_channels, kernel_size=self.kernel_size)) for _ in
            range(self.num_blocks)]

        def forward(x):
            x = self.residual(x)

            for attn, res in self.blocks:
                x = attn(x)
                x = res(x)

            return x


###############
# Base Blocks #
###############

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size of the embedding convolutional layers
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=self.kernel_size//2)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, padding=self.kernel_size//2)
        self.conv_con = nn.Conv2d(self.in_channels, self.out_channels, 1)

    def forward(self, x):
        assert x.shape[0] == self.in_channels
        z = self.conv_con(x)  # residual

        y = self.conv1(x)
        y = y.relu()
        y = self.conv2(y)
        y += z
        return y.relu()


class AttentionBlock(nn.Module):
    # Modeled after: https://github.com/gaozhihan/PreDiff/blob/main/src/prediff/taming/attention.py
    def __init__(self, channels, num_heads, rescale_output_factor: float = 1.0):
        """
        channels: number of channels
        num_heads: number of attention heads
        rescale_output_factor: factor for rescaling the output after self attention
        """
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.rescale_output_factor = rescale_output_factor

        self.q = nn.Linear(self.channels, self.channels)
        self.k = nn.Linear(self.channels, self.channels)
        self.v = nn.Linear(self.channels, self.channels)

        self.proj_attn = nn.Linear(self.channels, self.channels)

    def heads_to_batch_dim(self, tensor):
        batch_size, seq_length, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_length, self.num_heads, dim // self.num_heads)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_length, dim // self.num_heads)
        return tensor

    def batch_to_heads_dim(self, tensor):
        batch_size, seq_length, dim = tensor.shape
        tensor = tensor.reshape(batch_size // self.num_heads, self.num_heads, seq_length, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_length, dim * self.num_heads)
        return tensor

    def forward(self, x):
        residual = x
        batch, channel, height, width = x.shape

        # Calculate the query, key and value projections
        query = self.heads_to_batch_dim(self.q(x))
        key = self.heads_to_batch_dim(self.k(x))
        value = self.heads_to_batch_dim(self.v(x))

        scale = 1 / np.sqrt(self.channels, self.num_heads)

        attention_scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=scale
        )

        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
        x = torch.bmm(attention_probs, value)

        x = self.batch_to_heads_dim(x)

        x = self.proj_attn(x)

        # Reshape to original size
        x = x.transpose(-1, -2).reshape(batch, channel, height, width)

        # residual connection and rescaling
        return (x + residual) / self.rescale_output_factor
