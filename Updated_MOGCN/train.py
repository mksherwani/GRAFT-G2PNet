from data_loader import load_and_split_data  # remains the same
from train_mmae import train_mmae  # NEW training function for MMAE2Omics
from snf_module import build_snf_network, visualize_snf  # NEW SNF functions
from mmae_2omics import MMAE2Omics  # if needed for further operations
from models import MoGCN  # GCN model, remains mostly the same
import torch, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from extract_weights import extract_feature_importance  # Ensure you import the function properly


# Create output directory
os.makedirs('result', exist_ok=True)

# 2. Load data
(X_train_omic1, X_test_omic1, X_train_omic2, X_test_omic2, 
 y_train, y_test, train_idx, test_idx) = load_and_split_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Train the joint AE model. Adjust hyperparameters as needed.
trained_mmae = train_mmae(X_train_omic1, X_train_omic2, latent_dim=100, lr=0.001, batch_size=128, epochs=100, a=0.4, b=0.6, device=device)

# Load the final AE model
final_model = torch.load('model/MMAE2Omics_epoch_100.pkl', map_location=device)
final_model.eval()

top_features_omic1, top_features_omic2 = extract_feature_importance(
    'model/MMAE2Omics_epoch_100.pkl',
    omic1_features=list(X_train_omic1.columns),
    omic2_features=list(X_train_omic2.columns),
    omic1_std=X_train_omic1.std(),
    omic2_std=X_train_omic2.std(),
    topn=100
)
top_features_omic1.to_csv('result/topn_omic1.csv', index=False)
top_features_omic2.to_csv('result/topn_omic2.csv', index=False)

with torch.no_grad():
    # Convert training/test data to tensors
    train_tensor1 = torch.tensor(X_train_omic1.values, dtype=torch.float32).to(device)
    train_tensor2 = torch.tensor(X_train_omic2.values, dtype=torch.float32).to(device)
    test_tensor1  = torch.tensor(X_test_omic1.values, dtype=torch.float32).to(device)
    test_tensor2  = torch.tensor(X_test_omic2.values, dtype=torch.float32).to(device)
    
    # Extract separate latent outputs for each omic branch (ignore the combined one)
    _, latent_train1, latent_train2, _, _ = final_model(train_tensor1, train_tensor2)
    _, latent_test1,  latent_test2,  _, _ = final_model(test_tensor1, test_tensor2)


# Option: Use combined latent representation for further analysis or fusion.
# For SNF, you might want to use the separate latent representations:
latent_omic1_train = latent_train1.cpu().numpy()
latent_omic2_train = latent_train2.cpu().numpy()
sample_ids = list(X_train_omic1.index)


# 5. Build SNF network on fused features
print("Building SNF network...")
# Build the affinity network using the two latent representations from training data
snf_network = build_snf_network([latent_omic1_train, latent_omic2_train], metric='sqeuclidean', K=20, mu=0.5)
# Visualize and save the fused network
visualize_snf(snf_network, sample_ids, output_prefix='SNF')
# Convert SNF matrix to edge_index for your GCN, if needed:
def snf_to_edge_index(snf_matrix, threshold=0.7):
    rows, cols = np.where(snf_matrix > threshold)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return edge_index
edge_index = snf_to_edge_index(snf_network, threshold=0.7)

# 6. Prepare PyG Data object
print("Preparing graph data...")
# Concatenate or choose one of the representations. Here, we use the combined representation.
# Create feature tensors for each omics branch
x_omic1 = torch.tensor(np.vstack([latent_train1.cpu().numpy(), latent_test1.cpu().numpy()]), dtype=torch.float).to(device)
x_omic2  = torch.tensor(np.vstack([latent_train2.cpu().numpy(), latent_test2.cpu().numpy()]), dtype=torch.float).to(device)
y = torch.tensor(pd.concat([y_train, y_test]).astype(int).values, dtype=torch.long).to(device)
train_mask = torch.tensor([i in range(len(train_idx)) for i in range(len(y))], dtype=torch.bool).to(device)
test_mask = torch.tensor([i >= len(train_idx) for i in range(len(y))], dtype=torch.bool).to(device)
# Prepare the PyG Data object with separate fields for each branch
data = Data(
    x_omic1=x_omic1,
    x_omic2=x_omic2,
    edge_index=edge_index,  # assuming edge_index is built from your SNF module
    y=y,
    train_mask=train_mask,
    test_mask=test_mask
)

# 7. Initialize model
print("Initializing model...")
model = MoGCN(input_dims=[100, 100], hidden_dim=64, num_classes=4, dropout=0.5).to(device)
# Training loop here remains similar.

# 8. Training loop
print("Starting training...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_omic1, data.x_omic2, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(data.x_omic1, data.x_omic2, data.edge_index)
        val_loss = criterion(val_out[data.test_mask], data.y[data.test_mask])
        val_losses.append(val_loss.item())
        
        # Save best model
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

# Compute metrics
test_preds = preds[data.test_mask.cpu().numpy()]
test_labels = data.y[data.test_mask].cpu().numpy()
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
