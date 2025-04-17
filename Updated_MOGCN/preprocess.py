import torch
import torch.nn as nn
import numpy as np
from snf import snf
import seaborn as sns
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from snf import make_affinity, snf
from torch.utils.data import TensorDataset, DataLoader

class FusionAutoencoder(nn.Module):
    def __init__(self, input_dim1, input_dim2, latent_dim=128):
        super(FusionAutoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim1, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim2, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim1 + input_dim2)  # decoding both omics
        )

    def forward(self, omics):
        z1 = self.encoder1(omics[0])
        z2 = self.encoder2(omics[1])
        z = (z1 + z2) / 2
        out = self.decoder(z)
        return out

    def get_embedding(self, omics):
        z1 = self.encoder1(omics[0])
        z2 = self.encoder2(omics[1])
        return ((z1 + z2) / 2)
    

def train_fusion_ae(omics_data_list, epochs=100, batch_size=64, lr=1e-3):
    omic1, omic2 = omics_data_list

    # Convert to tensors
    tensor_omic1 = torch.tensor(omic1.values, dtype=torch.float32)
    tensor_omic2 = torch.tensor(omic2.values, dtype=torch.float32)

    # Assert shapes match
    assert tensor_omic1.shape[0] == tensor_omic2.shape[0], "Sample sizes do not match"

    # Create dataset & loader
    dataset = TensorDataset(tensor_omic1, tensor_omic2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer
    model = FusionAutoencoder(input_dim1=omic1.shape[1], input_dim2=omic2.shape[1], latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_omic1, batch_omic2 in loader:
            optimizer.zero_grad()
            recon_omic1, recon_omic2, _ = model([batch_omic1, batch_omic2])
            loss = criterion(recon_omic1, batch_omic1) + criterion(recon_omic2, batch_omic2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    return model


def build_snf_network(omics_list, K=20, mu=0.5):
    """Enhanced SNF matching original paper"""
    networks = []
    for omic in omics_list:
        # Original uses heat kernel similarity
        dist = pairwise_distances(omic, metric='cosine')
        W = np.exp(-dist**2 / (mu * np.mean(dist))**2)
        networks.append(W)
    
    # More sophisticated fusion in original
    fused = snf(networks, K=K, t=20)  # 20 iterations

    return fused

def snf_to_edge_index(snf_matrix, threshold=0.7):
    """Convert SNF matrix to PyG edge_index."""
    rows, cols = np.where(snf_matrix > threshold)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return edge_index
