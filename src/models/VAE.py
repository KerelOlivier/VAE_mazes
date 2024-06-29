"""
VAE template class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod, ABC


class IPrior(nn.Module):
    def sample(self, batch_size):
        """
        Sample from the prior distribution.

        Args:
            batch_size: int; Number of samples to generate
        """
        raise NotImplementedError

    def log_prob(self, z):
        """
        Compute the log probability of z under the prior.

        Args:
            z: torch.Tensor; Latent samples with shape (B,D)
        """
        raise NotImplementedError


class IEncoder(nn.Module):
    @staticmethod
    @abstractmethod
    def reparameterization(mu, log_var):
        """
        Reparameterization trick for VAEs.

        Args:
            mu: torch.Tensor; Mean of the distribution with shape (B,D)
            log_var: torch.Tensor; Log variance of the distribution with shape (B,D)
        """
        raise NotImplementedError
    
    @abstractmethod
    def encode(self, x, y=None):
        """
        Encode the input x into latent space.

        Args:
            x: torch.Tensor; Input tensor x with shape (B,D)
            y: torch.Tensor; Conditioning tensor y with shape (B,D) (optional)
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, x, y=None, return_components=False):
        """
        Sample from the variational posterior.

        Args:
            x: torch.Tensor; Input tensor x with shape (B,D)
            y: torch.Tensor; Conditioning tensor y with shape (B,D) (optional)
            return_components: bool; Whether to return the components of the sample (optional)
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_prob(self, x, y=None, return_components=False):
        """
        Compute the log probability of the input x under the encoder.

        Args:
            x: torch.Tensor; Input tensor x with shape (B,D)
            y: torch.Tensor; Conditioning tensor y with shape (B,D) (optional)
            return_components: bool; Whether to return the components of the log probability (optional)
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, x, y=None):
        """
        Forward pass of the encoder.

        Args:
            x: torch.Tensor; Input tensor x with shape (B,D)
            y: torch.Tensor; Conditioning tensor y with shape (B,D) (optional)
        """
        raise NotImplementedError
    

class IDecoder(nn.Module):
    @abstractmethod
    def sample(self, z, y=None):
        """
        Sample from the latent space.

        Args:
            z: torch.Tensor; Latent samples with shape (B,D)
            y: torch.Tensor; Conditioning tensor y with shape (B,D) (optional)
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_prob(self, x, z, y=None):
        """
        Compute the log probability of the input x under the decoder.

        Args:
            x: torch.Tensor; Input tensor x with shape (B,D)
            y: torch.Tensor; Conditioning tensor y with shape (B,D) (optional)
            z: torch.Tensor; Latent samples with shape (B,D)
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, x, z, y=None):
        """
        Forward pass of the decoder. Returns the log probability of x under the decoder.

        Args:
            x: torch.Tensor; Input tensor x with shape (B,D)
            y: torch.Tensor; Conditioning tensor y with shape (B,D) (optional)
            z: torch.Tensor; Latent samples with shape (B,D)
        """
        raise NotImplementedError


class VAE(nn.Module):
    def __init__(self, prior:IPrior, encoder:IEncoder, decoder:IDecoder, is_conditional=False, name="VAE"):
        super(VAE, self).__init__()
        """
        VAE template class.

        Args:
            prior: Prior; Prior distribution
            encoder: Encoder; Encoder network
            decoder: Decoder; Decoder network
            is_conditional: bool; Whether the VAE is conditional (optional)
                Checked in src/Trainer.py, to know whether to pass x & y to the encoder and decoder
                or only x
            name: str; Name of the VAE (optional), used in experiments
        """

        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.is_conditional = is_conditional
        self.name = name

    @torch.no_grad()
    def auto_encode(self, x, y=None):
        """
        Reconstruct input x, using the encoder/decoder.

        Args:
            x: torch.Tensor; input tensor x with shape (B,D)
            y: torch.Tensor; conditioning tensor y with shape (B,D) (optional)
        
        Returns:
            x_hat: torch.Tensor; reconstructed tensor x
        """
        z = self.encoder.sample(x=x, y=y)
        x_hat = self.decoder.sample(z=z, y=y)
        
        return x_hat

    def sample(self, y=None, batch_size=64):
        """
        Sample from the VAE.

        Args:
            y: torch.Tensor; conditioning tensor y with shape (B,D) (optional)
            batch_size: int; Number of samples to generate (optional)

        Returns:
            samples: torch.Tensor; Generated samples from p(x|z) (Decoder), given z ~ P(z) (Prior)
        """
        z = self.prior.sample(batch_size=batch_size)

        samples = self.decoder.sample(z=z, y=y)

        return samples

    def forward(self, x, y=None, reduction='mean', beta=1.0):
        """
        Compute the negative ELBO for the VAE as follows:

        L_NELBO = KL(q(z|x)||p(z)) - E_q(z|x)[log p(x|z)]

        Args:
            x: torch.Tensor; input tensor x with shape (B,D)
            y: torch.Tensor; conditioning tensor y with shape (B,D) (optional)
            reduction: str; Reduction type ('mean' or 'sum')

        Returns:
            NELBO loss: float; Negative ELBO loss of the VAE
        """
        enc_log_prob, z, mu, log_var = self.encoder.log_prob(x=x, y=y, return_components=True)


        reconstruction_loss = self.decoder.log_prob(x=x, y=y, z=z)
        
        kl_divergence = (enc_log_prob - self.prior.log_prob(z)).sum(-1)

        kl_divergence *= beta

        if reduction == 'sum':
            loss = (kl_divergence - reconstruction_loss).sum()
        elif reduction == 'mean':
            loss = (kl_divergence - reconstruction_loss).mean()
        else:
            raise ValueError('reduction type not supported')
        
        return loss
