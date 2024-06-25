"""
Transformer VAE model.

Combination of ResNet (down/up) blocks and Attention blocks.
"""
from src.models.VAE import VAE, IEncoder, IDecoder, IPrior

class TransformerEncoder(IEncoder):
    pass

class TransformerDecoder(IDecoder):
    pass