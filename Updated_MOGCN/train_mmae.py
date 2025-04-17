# train_mmae.py
import torch
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
import numpy as np
from mmae_2omics import MMAE2Omics

def train_mmae(omic1_data, omic2_data, latent_dim=100, lr=0.001, batch_size=32, epochs=100, a=0.4, b=0.3, device=torch.device('cpu')):
    """
    omic1_data, omic2_data: pandas DataFrame inputs (samples x features)
    """
    model = MMAE2Omics(omic1_data.shape[1], omic2_data.shape[1], latent_dim, a, b)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # Combine the two omics data into a dataset
    tensor_omic1 = torch.tensor(omic1_data.values, dtype=torch.float32)
    tensor_omic2 = torch.tensor(omic2_data.values, dtype=torch.float32)
    dummy_target1 = tensor_omic1  # if using reconstruction loss
    dummy_target2 = tensor_omic2

    dataset = Data.TensorDataset(tensor_omic1, tensor_omic2, dummy_target1, dummy_target2)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            x1, x2, target1, target2 = [b.to(device) for b in batch]
            optimizer.zero_grad()
            combined, latent1, latent2, decoded1, decoded2 = model(x1, x2)
            # Compute reconstruction loss for both omics branches
            loss1 = criterion(decoded1, target1)
            loss2 = criterion(decoded2, target2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        # Save the model at every 10 epochs for weight extraction later
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model, f'model/MMAE2Omics_epoch_{epoch}.pkl')
    return model

# Example usage:
# trained_model = train_mmae(omic1_train_df, omic2_train_df, latent_dim=100, lr=0.001, batch_size=32, epochs=100, a=0.4, b=0.6, device=device)
