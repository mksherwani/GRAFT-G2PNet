# mmae_2omics.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMAE2Omics(nn.Module):
    def __init__(self, in_dim1, in_dim2, latent_dim=100, a=0.4, b=0.3):
        """
        in_dim1, in_dim2: dimensions of omics inputs
        latent_dim: dimension of each omics’ latent representation
        a, b: weights for each omics branch (they must sum to 1)
        """
        super(MMAE2Omics, self).__init__()
        self.a = a  # weight for omics 1
        self.b = b  # weight for omics 2
        
        # Encoder for omic 1
        self.encoder_omics1 = nn.Sequential(
            nn.Linear(in_dim1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Encoder for omic 2
        self.encoder_omics2 = nn.Sequential(
            nn.Linear(in_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # If you need decoders for reconstruction, define them here:
        self.decoder_omics1 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim1)
        )
        self.decoder_omics2 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim2)
        )

    def forward(self, x1, x2):
        # Get latent representations for each omic branch.
        latent1 = self.encoder_omics1(x1)
        latent2 = self.encoder_omics2(x2)
        # Optionally, combine them according to their weights.
        # For example, a weighted concatenation:
        combined = torch.cat([self.a * latent1, self.b * latent2], dim=1)
        # For reconstruction, you could decode separately:
        decoded1 = self.decoder_omics1(latent1)
        decoded2 = self.decoder_omics2(latent2)
        return combined, latent1, latent2, decoded1, decoded2
