# mmae2omics_full.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import pandas as pd
import numpy as np

# MMAE2Omics Model Definition
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
        # Decoder for omic 1
        self.decoder_omics1 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim1)
        )
        # Decoder for omic 2
        self.decoder_omics2 = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim2)
        )

    def forward(self, x1, x2):
        # Get latent representations for each omic branch
        latent1 = self.encoder_omics1(x1)
        latent2 = self.encoder_omics2(x2)
        # Combine latent features
        combined = torch.cat([self.a * latent1, self.b * latent2], dim=1)
        # Decode separately
        decoded1 = self.decoder_omics1(latent1)
        decoded2 = self.decoder_omics2(latent2)
        return combined, latent1, latent2, decoded1, decoded2

# Training Function
def train_mmae(omic1_data, omic2_data, latent_dim=100, lr=0.001, batch_size=32, epochs=100, a=0.4, b=0.3, device=torch.device('cpu')):
    """
    omic1_data, omic2_data: pandas DataFrame inputs (samples x features)
    """
    model = MMAE2Omics(omic1_data.shape[1], omic2_data.shape[1], latent_dim, a, b)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # Prepare dataset
    tensor_omic1 = torch.tensor(omic1_data.values, dtype=torch.float32)
    tensor_omic2 = torch.tensor(omic2_data.values, dtype=torch.float32)
    dummy_target1 = tensor_omic1
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
            loss1 = criterion(decoded1, target1)
            loss2 = criterion(decoded2, target2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        # Save model checkpoint
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model, f'model/MMAE2Omics_epoch_{epoch}.pkl')
    return model

# Feature Importance Extraction Function
def extract_feature_importance(model_path, omic1_features, omic2_features, omic1_std, omic2_std, topn=100):
    model = torch.load(model_path, map_location='cpu')
    state = model.state_dict()
    # Extract weight matrices
    weight_omic1 = state['encoder_omics1.0.weight'].detach().cpu().numpy()
    weight_omic2 = state['encoder_omics2.0.weight'].detach().cpu().numpy()
    
    # Calculate importance
    importance_omic1 = np.sum(np.abs(weight_omic1), axis=0) * omic1_std.values
    importance_omic2 = np.sum(np.abs(weight_omic2), axis=0) * omic2_std.values
    
    # Create DataFrames
    df_omic1 = pd.DataFrame({
        'Feature': omic1_features,
        'Importance': importance_omic1
    })
    df_omic2 = pd.DataFrame({
        'Feature': omic2_features,
        'Importance': importance_omic2
    })

    # Select top features
    top_omic1 = df_omic1.nlargest(topn, 'Importance')
    top_omic2 = df_omic2.nlargest(topn, 'Importance')
    
    return top_omic1, top_omic2

# Example usage (commented out):
# trained_model = train_mmae(omic1_train_df, omic2_train_df, latent_dim=100, lr=0.001, batch_size=32, epochs=100, a=0.4, b=0.6, device=device)
# top_features_omic1, top_features_omic2 = extract_feature_importance('model/MMAE2Omics_epoch_10.pkl', 
#     omic1_features=list(omic1_train_df.columns),
#     omic2_features=list(omic2_train_df.columns),
#     omic1_std=omic1_train_df.std(),
#     omic2_std=omic2_train_df.std(),
#     topn=100)
# top_features_omic1.to_csv('result/topn_omic1.csv', index=False)
# top_features_omic2.to_csv('result/topn_omic2.csv', index=False)
