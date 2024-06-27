"""
Transformer VAE model.

Combination of ResNet (down/up) blocks and Attention blocks.
"""
import torch

from src.models.VAE import IEncoder, IDecoder
from torch import nn
from torch.nn.functional import interpolate
from src.utils.auxiliary import log_normal_diag, log_bernoulli
import numpy as np


#################
# Encoder Block #
#################

class TransformerEncoder(IEncoder):
    def __init__(self,
                 input_dim=(1, 129, 129),
                 out_channels=(64,),
                 layers_per_block=2,
                 kernel_size=3,
                 down_sample_factor=2,
                 num_heads=16):
        super().__init__()
        """
        Encoder part for the transformer VAE
        
        :param input_dim: input dimension of the image.(channels, width, height)
        :param out_channels: output channels for each block
        :param layers_per_block: number of residual blocks per encoder block
        :param kernel_size: kernel size for the residual blocks
        :param down_sample_factor: how fast to down sample between each step
        :param num_heads: number of heads for multi-head attention    
        """
        self.in_dim = input_dim
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block

        # Create the encoder blocks
        self.blocks = []
        input_channel = self.in_dim[0]
        for out_channel in self.out_channels:
            self.blocks.append(TransformerEncoderBlock(input_channel,
                                                       out_channel,
                                                       layers_per_block,
                                                       kernel_size,
                                                       down_sample_factor))
            input_channel = out_channel
        self.blocks = nn.ModuleList(self.blocks)
        # Create the attention middle block
        self.middle = TransformerMidBlock(out_channels[-1],
                                          out_channels[-1],
                                          1,
                                          num_heads,
                                          down_sample_factor,
                                          kernel_size)

    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(input=mu)

        return mu + std * eps

    def encode(self, x, y=None):
        # Add conditional
        if y is not None:
            x += y

        # Encoder
        for block in self.blocks:
            x = block(x)

        # Middle block
        x = self.middle(x)

        # Flatten for ease of use
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # Split output into mu an log_var tensors
        mu, log_var = x.chunk(2, dim=1)
        return mu, log_var

    def sample(self, x, y=None, return_components=False):
        mu, log_var = self.encode(x, y)

        z = self.reparameterization(mu, log_var)

        if return_components:
            return z, mu, log_var
        return z

    def log_prob(self, x, y=None, return_components=False):
        z, mu, log_var = self.sample(x, return_components, True)

        if return_components:
            return log_normal_diag(z, mu, log_var), z, mu, log_var
        return log_normal_diag(z, mu, log_var)

    def forward(self, x, y=None):
        return self.log_prob(x, y)


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers: int = 1,
                 kernel_size: int = 3,
                 down_sample_factor=2.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.kernel_size = kernel_size
        self.down_sample_factor = down_sample_factor

        self.res1 = ResidualBlock(in_channels, out_channels, kernel_size)
        self.blocks = nn.ModuleList(
            [ResidualBlock(out_channels, self.out_channels, kernel_size) for _ in range(layers - 1)])
        self.down_sample = DownSample2D(2)

    def forward(self, x):
        x = self.res1(x)
        for block in self.blocks:
            x = block(x)
        return self.down_sample(x)


class DownSample2D(nn.Module):
    def __init__(self, window_size=2):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size=window_size, stride=window_size)

    def forward(self, x):
        return self.pooling(x)


#################
# Decoder block #
#################

class TransformerDecoder(IDecoder):
    def __init__(self,
                 input_shape: (int, int, int),
                 output_shapes: ((int, int, int),),
                 conditional_shape: (int, int, int),
                 layers_per_block=2,
                 kernel_size=3,
                 ):
        """
        :param input_shape: expected input shape of decoder, (channels, height, width)
        :param conditional_shape: shape of the conditional, (channels, height, width)
        each decoder block, last one must have the number of channels of your output
        :param layers_per_block: number of residual blocks per decoder block
        :param kernel_size: kernel size of the decoder
        """
        super().__init__()

        self.input_shape = input_shape

        # Conditional linear projection
        in_size = conditional_shape[0] * conditional_shape[1] * conditional_shape[2]
        out_size = input_shape[0] * input_shape[1] * input_shape[2] // 2

        self.con_lin = nn.Linear(in_size, out_size)

        # Create Decoder blocks
        self.blocks = []

        in_shape = (self.input_shape[0]//2, self.input_shape[1], self.input_shape[2])
        for shape in output_shapes:
            self.blocks.append(DecoderBlock(in_channels=in_shape[0],
                                            out_channels=shape[0],
                                            layers=layers_per_block,
                                            kernel_size=kernel_size,
                                            output_shape=(shape[1], shape[2])
                                            ))
            in_shape = shape
        self.blocks = nn.ModuleList(self.blocks)

    def decode(self, z, y=None):
        # Add conditional
        if y is not None:
            # reshape
            y = torch.flatten(y, 1)
            tmp = self.con_lin(y)
            z += tmp

        # Reshape z
        z = torch.reshape(z, (z.shape[0], self.input_shape[0]//2, self.input_shape[1], self.input_shape[2]))



        # Run decoder
        for block in self.blocks:
            z = block(z)

        z = torch.sigmoid(z)
        return z

    def sample(self, z, y=None):
        mu = self.decode(z, y)
        x_new = torch.bernoulli(mu)

        return x_new

    def log_prob(self, x, z, y=None):
        mu = self.decode(z, y)
        return log_bernoulli(x, mu, reduction='sum')

    def forward(self, z, y=None):
        return self.log_prob(z, y)


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers: int = 1,
                 kernel_size: int = 3,
                 output_shape=(129, 129)):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param layers: number of residual blocks
        :param kernel_size: kernel size for the residual blocks
        :param output_shape: shape of the up sampled output (width, height)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.kernel_size = kernel_size
        self.output_shape = output_shape

        self.res1 = ResidualBlock(self.in_channels, self.out_channels, self.kernel_size)
        self.blocks = nn.ModuleList(
            [ResidualBlock(self.out_channels, self.out_channels, self.kernel_size) for _ in range(layers - 1)])
        self.up_sample = UpSample2D(self.out_channels, output_shape=self.output_shape, kernel_size=3)

    def forward(self, x):
        x = self.res1(x)
        for block in self.blocks:
            x = block(x)
        return self.up_sample(x)


class UpSample2D(nn.Module):
    # using solutions to checkerboard artifacts discussed here: https://distill.pub/2016/deconv-checkerboard/
    def __init__(self, channels, output_shape=(129, 129), kernel_size=3):
        """
        Up sample 2D convolution
        :param output_shape: dimensions to which to up sample the input
        """
        super().__init__()
        self.output_shape = output_shape

        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = interpolate(x, size=self.output_shape, mode='bilinear')
        x = self.conv(x)
        return x


##################
# UNet mid block #
##################

class TransformerMidBlock(nn.Module):
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
        self.att_blocks = nn.ModuleList([
            AttentionBlock(self.out_channels, self.num_heads, rescale_output_factor=self.rescale_output_factor) for _ in
            range(self.num_blocks)])
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.out_channels, self.out_channels, kernel_size=self.kernel_size) for _ in
            range(self.num_blocks)])

        self.blocks = zip(self.att_blocks, self.res_blocks)

    def forward(self, x):
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
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=self.kernel_size // 2)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, padding=self.kernel_size // 2)
        self.conv_con = nn.Conv2d(self.in_channels, self.out_channels, 1)

    def forward(self, x):
        assert x.shape[1] == self.in_channels
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

        x = x.view(batch, channel, width*height)

        # Calculate the query, key and value projections
        query = self.heads_to_batch_dim(self.q(x))
        key = self.heads_to_batch_dim(self.k(x))
        value = self.heads_to_batch_dim(self.v(x))

        scale = 1 / np.sqrt(self.channels/ self.num_heads)

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
