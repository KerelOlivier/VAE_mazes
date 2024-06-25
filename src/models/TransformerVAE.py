"""
Transformer VAE model.

Combination of ResNet (down/up) blocks and Attention blocks.
"""
from src.models.VAE import VAE, IEncoder, IDecoder, IPrior


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
    def __init__(self):
        super().__init__()

###############
# Base Blocks #
###############

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
