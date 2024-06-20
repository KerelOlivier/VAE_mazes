import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.VAE import IPrior
from src.utils import log_normal_diag


class MogPrior(IPrior):
    """Mixture of Gaussians prior, adapted from https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_priors_example.ipynb"""
    def __init__(self, latent_dim, num_components):
        """
        Mixtures of Gaussians prior for the VAE.

        Args:
            latent_dim: int; Number of latent dimensions
            num_components: int; Number of components in the mixture
        """
        super(MogPrior, self).__init__()

        self.latent_dim = latent_dim
        self.num_components = num_components

        # params
        self.means = nn.Parameter(torch.randn(num_components, self.latent_dim))
        self.logvars = nn.Parameter(torch.randn(num_components, self.latent_dim))

        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_components, 1, 1))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_params(self):
        """Return parameters of the Gaussian components."""
        return self.means, self.logvars

    def sample(self, batch_size):
        """
        Sample z from the prior.

        Args:
            batch_size: int; Batch size

        Returns:
            z: torch.Tensor; Samples from the prior
        """
        # mu, logvar
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and logvars
        eps = torch.randn(batch_size, self.latent_dim).to(self.device)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])
            else:
                z = torch.cat((z, means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])), 0)
        return z

    def log_prob(self, z):
        """
        Compute the log probability ln p(z) for given z.
        """
        # mu, lof_var
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0) # 1 x B x L
        means = means.unsqueeze(1) # K x 1 x L
        logvars = logvars.unsqueeze(1) # K x 1 x L

        log_p = log_normal_diag(z, means, logvars) + torch.log(w) # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L

        return log_prob