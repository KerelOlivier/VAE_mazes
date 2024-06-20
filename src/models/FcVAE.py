from src.models.VAE import IEncoder, IDecoder
from src.utils.auxiliary import log_normal_diag, log_bernoulli

import torch
import torch.nn as nn


class FcEncoder(IEncoder):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Fully connected encoder for VAE.

        Args:
            input_dim: int; input dimension
            hidden_dims: List[int]; hidden dimensions
            latent_dim: int; latent dimension
        """
        super(FcEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Initialize layers
        self.layers = []
        self.make_layers()

        self.to_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.to_log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def make_layers(self):
        """
        Make layers for the encoder.
        """
        # Initialize input size
        input_size = self.input_dim
        for h_dim in self.hidden_dims:
            self.layers.append(nn.Linear(input_size, h_dim))
            self.layers.append(nn.SiLU())
            input_size = h_dim

        # Sequential model
        self.forward_net = nn.Sequential(*self.layers)

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
            x: torch.Tensor; Input tensor x with shape (B,D)
            y: torch.Tensor; Conditioning tensor y with shape (B,D) (optional)

        Returns:
            mu: torch.Tensor; mean of the distribution with shape (B,L)
            log_var: torch.Tensor; log variance of the distribution with shape (B,L)
        """
        if y is not None:
            x = torch.cat([x, y], dim=-1)
        
        h = self.forward_net(x)
        mu = self.to_mu(h)
        log_var = self.to_log_var(h)
        return mu, log_var

    def sample(self, x, y=None, return_components=False):
        """
        Sample from the variational posterior q(z|x)

        Args:
            x: torch.Tensor; input tensor x with shape (B,D)
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
            x: torch.Tensor; input tensor x with shape (B,D)
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
            x: torch.Tensor; input tensor x with shape (B,D)

        Returns:
            log_p: torch.Tensor; log probability - log q(z|x)
        """
        return self.log_prob(x, y)
    
class FcDecoder(IDecoder):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        """
        Fully connected decoder for VAE.

        Args:
            latent_dim: int; latent dimension
            hidden_dims: List[int]; hidden dimensions
            output_dim: int; output dimension
        """
        super(FcDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Initialize layers
        self.layers = []
        self.make_layers()

        self.to_output = nn.Linear(hidden_dims[-1], output_dim)

    def make_layers(self):
        """
        Make layers for the decoder.
        """
        # Initialize input size
        input_size = self.latent_dim
        for h_dim in self.hidden_dims:
            self.layers.append(nn.Linear(input_size, h_dim))
            self.layers.append(nn.SiLU())
            input_size = h_dim

        # Sequential model
        self.forward_net = nn.Sequential(*self.layers)

    def decode(self, z, y=None):
        """
        Decode the latent variable z into the output space.

        Args:
            z: torch.Tensor; latent variable z
            y: torch.Tensor; conditioning tensor y

        Returns:
            x: torch.Tensor; output tensor x
        """
        if y is not None:
            z = torch.cat([z, y], dim=-1)
        
        x = self.forward_net(z)
        x = self.to_output(x)
        mu = torch.sigmoid(x)
        return mu

    def sample(self, z, y=None):
        """
        Sample from the decoder.

        Args:
            z: torch.Tensor; latent samples with shape (B,D)
            y: torch.Tensor; conditioning tensor y with shape (B,D) (optional)

        Returns:
            x: torch.Tensor; output tensor x
        """
        mu = self.decode(z, y)
        x_new = torch.bernoulli(mu)

        return x_new
    
    def log_prob(self, x, z, y=None):
        """
        Compute the log probability of the decoder.

        Args:
            x: torch.Tensor; input tensor x with shape (B,D)
            z: torch.Tensor; latent samples with shape (B,D)
            y: torch.Tensor; conditioning tensor y with shape (B,D) (optional)

        Returns:
            log_p: torch.Tensor; log probability - log p(x|z)
        """
        mu = self.decode(z, y)
        return log_bernoulli(x, mu, reduction='sum')
    
    def forward(self, z, y=None):
        """
        Compute the log probability of the decoder.

        Args:
            z: torch.Tensor; latent variable z
            y: torch.Tensor; conditioning tensor y

        Returns:
            log_p: torch.Tensor; log probability - log p(x|z)
        """
        return self.log_prob(z, y)