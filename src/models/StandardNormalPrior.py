import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.VAE import IPrior
from src.utils import log_standard_normal


class StandardNormalPrior(IPrior):
    """Standard Gaussian Prior"""
    def __init__(self, latent_dim):
        """
        Standard Normal Prior for the VAE.

        Args:
            latent_dim: int; Number of latent dimensions
        """
        super(StandardNormalPrior, self).__init__()

        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # params weights
        self.means = torch.zeros(1, latent_dim)
        self.logvars = torch.zeros(1, latent_dim)

    def get_params(self):
        """
        Get the parameters of the prior.

        Returns:
            means: torch.Tensor; Means of the prior
            logvars: torch.Tensor; Logvars of the prior
        """
        return self.means, self.logvars

    def sample(self, batch_size):
        """
        Sample z from the prior.

        Args:
            batch_size: int; Batch size
        
        Returns:
            z: torch.Tensor; Samples from the prior
        """
        return torch.randn(batch_size, self.latent_dim).to(self.device)
    
    def log_prob(self, z):
        """
        Compute the log probability of the input z under the prior.

        Args:
            z: torch.Tensor; Input tensor z with shape (B,D)

        Returns:
            log_p: torch.Tensor; log probability - log p(z)
        """
        return log_standard_normal(z)