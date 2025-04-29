#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, random
import torch
import numpy as np
import pandas as pd
from data_loader import load_and_split_data
from snf_module import build_snf_network, visualize_snf
from mmae_2omics import train_mmae, MMAE2Omics, extract_feature_importance
from models import MoGCN
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create output directory
os.makedirs('result', exist_ok=True)

# 2. Load data
X_train_omic1, X_test_omic1, X_train_omic2, X_test_omic2, \
    y_train, y_test, train_idx, test_idx = load_and_split_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Train the joint AE model
trained_mmae = train_mmae(
    X_train_omic1, X_train_omic2,
    latent_dim=100,
    lr=0.001,
    batch_size=128,
    epochs=100,
    a=0.4,
    b=0.6,
    device=device
)

# Load and set final AE model
default_model_path = 'model/MMAE2Omics_epoch_100.pkl'
final_model = torch.load(default_model_path, map_location=device)
final_model.eval()

# 4. Extract latent representations
with torch.no_grad():
    train_tensor1 = torch.tensor(X_train_omic1.values, dtype=torch.float32).to(device)
    train_tensor2 = torch.tensor(X_train_omic2.values, dtype=torch.float32).to(device)
    test_tensor1  = torch.tensor(X_test_omic1.values,  dtype=torch.float32).to(device)
    test_tensor2  = torch.tensor(X_test_omic2.values,  dtype=torch.float32).to(device)

    _, latent_train1, latent_train2, _, _ = final_model(train_tensor1, train_tensor2)
    _, latent_test1,  latent_test2,  _, _ = final_model(test_tensor1,  test_tensor2)

# --- Save latent features ---
latent_train = torch.cat([latent_train1, latent_train2], dim=1).cpu().numpy()
latent_test  = torch.cat([latent_test1,  latent_test2 ], dim=1).cpu().numpy()

df_latent_train = pd.DataFrame(latent_train, index=X_train_omic1.index)
df_latent_test  = pd.DataFrame(latent_test,  index=X_test_omic1.index)

df_latent_train.to_csv('result/latent_train.csv', header=True)
df_latent_test.to_csv('result/latent_test.csv',   header=True)

# --- Epoch-wise feature importance ---
omic1_feats = list(X_train_omic1.columns)
omic2_feats = list(X_train_omic2.columns)
omic1_std   = X_train_omic1.std()
omic2_std   = X_train_omic2.std()
topn_omic1_df = pd.DataFrame()
topn_omic2_df = pd.DataFrame()

# build list of saved epochs
epoch_list = list(range(10, 101, 10))
if 100 % 10 != 0:
    epoch_list.append(100)

for epoch in epoch_list:
    model_path = f'model/MMAE2Omics_epoch_{epoch}.pkl'
    t1, t2 = extract_feature_importance(
        model_path,
        omic1_features=omic1_feats,
        omic2_features=omic2_feats,
        omic1_std=omic1_std,
        omic2_std=omic2_std,
        topn=100
    )
    topn_omic1_df[f'epoch_{epoch}'] = t1['Feature'].values
    topn_omic2_df[f'epoch_{epoch}'] = t2['Feature'].values

# save epoch-wise feature lists
topn_omic1_df.to_csv('result/topn_omic1_epochwise.csv', index=False)
topn_omic2_df.to_csv('result/topn_omic2_epochwise.csv', index=False)

# 5. Build SNF network on fused features
print("Building SNF network...")
snf_network = build_snf_network(
    [latent_train1.cpu().numpy(), latent_train2.cpu().numpy()],
    metric='sqeuclidean', K=20, mu=0.5
)
visualize_snf(snf_network, list(X_train_omic1.index), output_prefix='SNF')

def snf_to_edge_index(snf_matrix, threshold=0.7):
    rows, cols = np.where(snf_matrix > threshold)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)

edge_index = snf_to_edge_index(snf_network, threshold=0.7)

# 6. Prepare PyG Data object
print("Preparing graph data...")
x_omic1 = torch.tensor(np.vstack(
    [latent_train1.cpu().numpy(), latent_test1.cpu().numpy()]
), dtype=torch.float).to(device)
x_omic2 = torch.tensor(np.vstack(
    [latent_train2.cpu().numpy(), latent_test2.cpu().numpy()]
), dtype=torch.float).to(device)
y = torch.tensor(
    pd.concat([y_train, y_test]).astype(int).values,
    dtype=torch.long
).to(device)
train_mask = torch.tensor(
    [i in range(len(train_idx)) for i in range(len(y))],
    dtype=torch.bool
).to(device)
test_mask = torch.tensor(
    [i >= len(train_idx) for i in range(len(y))],
    dtype=torch.bool
).to(device)

data = Data(
    x_omic1=x_omic1,
    x_omic2=x_omic2,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    test_mask=test_mask
)

# 7. Initialize GCN model
print("Initializing model...")
model = MoGCN(input_dims=[100, 100], hidden_dim=64, num_classes=4, dropout=0.5).to(device)

# 8. Training loop
print("Starting training...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_omic1, data.x_omic2, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        val_out = model(data.x_omic1, data.x_omic2, data.edge_index)
        val_loss = criterion(val_out[data.test_mask], data.y[data.test_mask])
        val_losses.append(val_loss.item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'result/best_model.pt')
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.savefig('result/training_curves.png')
plt.close()

# 9. Evaluation with best model
print("Evaluating best model...")
model.load_state_dict(torch.load('result/best_model.pt'))
model.eval()
with torch.no_grad():
    logits = model(data.x_omic1, data.x_omic2, data.edge_index)
    preds = logits.argmax(dim=1).cpu().numpy()

test_preds = preds[test_mask.cpu().numpy()]
test_labels = data.y[test_mask].cpu().numpy()
accuracy = accuracy_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds, average='weighted')

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('result/confusion_matrix.png')
plt.close()

# Save predictions with sample IDs
results_df = pd.DataFrame({
    'sample_id': test_idx,
    'true_label': test_labels,
    'predicted_label': test_preds
})
results_df.to_csv('result/test_predictions.csv', index=False)

print("\nTraining complete! Results saved in 'result' directory")
