from src.models.VAE import IEncoder, IDecoder
from src.utils.auxiliary import log_normal_diag, log_bernoulli

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, activation=True, num_groups=4):
        """
        Convolutional block with convolutional layer, batch normalization, and activation.

        Args:
            in_channels: int; Number of input channels
            out_channels: int; Number of output channels
            kernel_size: int; Kernel size of the convolutional layer
            stride: int; Stride of the convolutional layer
            activation: bool; Flag to include activation function
        """
        super(ConvBlock, self).__init__()
        if num_groups > out_channels:
            num_groups = out_channels
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.norm(x)
        if self.activation:
            x = F.silu(x)
        return x

class ConvDownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers = 2):
        """
        Convolutional down block.

        Args:
            in_channels: int; Number of input channels
            out_channels: int; Number of output channels
            num_layers: int; Number of convolutional layers
        """
        super(ConvDownBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, activation=True))
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        return x
    
class ConvUpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers = 2, size = None, scale_factor = 2):
        """
        Convolutional up block.

        Args:
            in_channels: int; Number of input channels
            out_channels: int; Number of output channels
            num_layers: int; Number of convolutional layers
            size: Tuple[int]; Size of the output tensor (optional)
            scale_factor: int; Scale factor for upsampling (optional)
        """
        super(ConvUpBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, activation=True))
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        if size is None:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Upsample(size=[size[2], size[3]], mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
        x = self.layers(x)
        return x

class ConvEncoder(IEncoder):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Convolutional encoder for VAE.

        Args:
            input_dim: int; input dimension
            hidden_dims: List[int]; hidden dimensions
            latent_dim: int; latent dimension
        """
        super(ConvEncoder, self).__init__()
        self.input_dim = input_dim #(B,C,H,W)
        self.hidden_dims = hidden_dims # [Ci, Cj, ...]
        self.latent_dim = latent_dim # L

        # Initialize layers
        self.layers = []
        self.make_layers()

        self.to_mu = nn.Sequential(ConvBlock(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, activation=False),
                                   nn.Flatten())
        self.to_log_var = nn.Sequential(ConvBlock(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, activation=False),
                                   nn.Flatten())

    def make_layers(self):
        """
        Make layers for the encoder.
        """
        # Initialize input size
        input_size = self.hidden_dims[0]
        for h_dim in self.hidden_dims:
            self.layers.append(ConvDownBlock(input_size, h_dim))
            self.layers.append(nn.Dropout(p=0.2))
            input_size = h_dim

        # Sequential model
        self.forward_net = nn.Sequential(*self.layers)

        self.in_net = nn.Sequential(
            ConvBlock(1, self.hidden_dims[0], kernel_size=3, stride=1, activation=True),
            nn.Dropout(p=0.2)
        )

        self.conditional_net = nn.Sequential(
            ConvBlock(1, self.hidden_dims[0], kernel_size=3, stride=1, activation=False),
            nn.Dropout(p=0.2)
        )

    @staticmethod
    def reparameterization(mu, log_var):
        """
        Reparameterization trick. Given mu, log_var; convert log_var to standard deviation by taking the root of the exponent.
        Sample random noise epsilon, then multiply by the standard deviation and add onto mu to get z.

        Args:
            mu: torch.Tensor; means of the diagonal gaussian distribution q(z|x)
            log_var: torch.Tensor; log variance of q(z|x)

        Returns:
            z: torch.Tensor; latent variable z
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(input=mu)
        return mu + std * eps
    
    def encode(self, x, y=None):
        """
        Encode the input x into latent space.

        Args:
            x: torch.Tensor; Input tensor x with shape (B,C,H,W)
            y: torch.Tensor; Conditioning tensor y with shape (B,C,H,W) (optional)

        Returns:
            mu: torch.Tensor; mean of the distribution with shape (B,L)
            log_var: torch.Tensor; log variance of the distribution with shape (B,L)
        """
        x = self.in_net(x)

        if y is not None:
            y = self.conditional_net(y)
            x = x + y
        
        h = self.forward_net(x)
        mu = self.to_mu(h)
        log_var = self.to_log_var(h)
        return mu, log_var

    def sample(self, x, y=None, return_components=False):
        """
        Sample from the variational posterior q(z|x)

        Args:
            x: torch.Tensor; input tensor x with shape (B,C,H,W)
            return_components: bool; flag to return mu, log var

        Returns:
            z: torch.Tensor; latent variable z
            (optional) mu: torch.Tensor; means of the diagonal gaussian distribution q(z|x)
            (optional) log_var: torch.Tensor; log variance of q(z|x)
        """
        mu, log_var = self.encode(x, y)

        z = self.reparameterization(mu, log_var)

        if return_components:
            return z, mu, log_var
        return z
    
    def log_prob(self, x, y=None, return_components=False):
        """
        Compute the log probability of the variational posterior

        Args:
            x: torch.Tensor; input tensor x with shape (B,C,H,W)
            return_components: bool; flag to return z, mu, log var

        Returns:
            log_p: torch.Tensor; log probability - log q(z|x)
            (optional) z: torch.Tensor; latent z
            (optional) mu: torch.Tensor; means of the diagonal gaussian distribution q(z|x)
            (optional) log_var: torch.Tensor; log variance of q(z|x)
        """
        z, mu, log_var = self.sample(x, y, return_components=True)

        if return_components:
            return log_normal_diag(z, mu, log_var), z, mu, log_var
        return log_normal_diag(z, mu, log_var)
    
    def forward(self, x, y=None):
        """
        Compute the log probability of the variational posterior q(z|x)

        Args:
            x: torch.Tensor; input tensor x with shape (B,C,H,W)

        Returns:
            log_p: torch.Tensor; log probability - log q(z|x)
        """
        return self.log_prob(x, y)
    
class ConvDecoder(IDecoder):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        """
        Convolutional decoder for VAE.

        Args:
            latent_dim: int; latent dimension
            hidden_dims: List[int]; hidden dimensions
            output_dim: int; output dimension
        """
        super(ConvDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Initialize layers
        self.layers = []
        self.make_layers()

        self.to_output_size = nn.Sequential(
            ConvUpBlock(hidden_dims[-1], hidden_dims[-1], num_layers=2, scale_factor=None, size=self.output_dim),
            ConvBlock(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, activation=True)
        )

        self.to_output = nn.Sequential(
            ConvBlock(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=1, activation=True),
            ConvBlock(hidden_dims[-1], 1, kernel_size=3, stride=1, activation=False)
        )

        self.after_encoder_size = (self.hidden_dims[0], output_dim[2] // 2**len(self.hidden_dims), output_dim[2] // 2**len(self.hidden_dims))

        self.final_sample = torch.bernoulli

    def make_layers(self):
        """
        Make layers for the decoder.
        """
        # Initialize input size
        input_size = self.hidden_dims[0]
        for i, h_dim in enumerate(self.hidden_dims):
            self.layers.append(ConvUpBlock(input_size, h_dim))
            input_size = h_dim

        # Sequential model
        self.forward_net = nn.Sequential(*self.layers)

        self.conditional_net = nn.Sequential(
            ConvBlock(self.output_dim[1], self.output_dim[1], kernel_size=3, stride=1, activation=True),
            nn.Dropout(p=0.2)
        )

    def decode(self, z, y=None):
        """
        Decode the latent variable z into the output space.

        Args:
            z: torch.Tensor; latent variable z
            y: torch.Tensor; conditioning tensor y

        Returns:
            x: torch.Tensor; output tensor x
        """
        z = z.reshape(z.shape[0], self.after_encoder_size[0], self.after_encoder_size[1], self.after_encoder_size[2])
        z = self.forward_net(z)
        z = self.to_output_size(z)
        if y is not None:
            y = self.conditional_net(y)
            z = z + y
        
        z = self.to_output(z)

        mu = torch.sigmoid(z)
        return mu

    def sample(self, z, y=None):
        """
        Sample from the decoder.

        Args:
            z: torch.Tensor; latent samples with shape (B,C,H,W)
            y: torch.Tensor; conditioning tensor y with shape (B,C,H,W) (optional)

        Returns:
            x: torch.Tensor; output tensor x
        """
        mu = self.decode(z, y)
        x_new = self.final_sample(mu)

        return x_new
    
    def log_prob(self, x, z, y=None):
        """
        Compute the log probability of the decoder.

        Args:
            x: torch.Tensor; input tensor x with shape (B,C,H,W)
            z: torch.Tensor; latent samples with shape (B,C,H,W)
            y: torch.Tensor; conditioning tensor y with shape (B,C,H,W) (optional)

        Returns:
            log_p: torch.Tensor; log probability - log p(x|z)
        """
        mu = self.decode(z, y)
        return log_bernoulli(x, mu, reduction='sum')
    
    def forward(self, x, z, y=None):
        """
        Compute the log probability of the decoder.

        Args:
            z: torch.Tensor; latent variable z
            y: torch.Tensor; conditioning tensor y

        Returns:
            log_p: torch.Tensor; log probability - log p(x|z)
        """
        return self.log_prob(x, z, y)