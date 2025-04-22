import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
import matplotlib.pyplot as plt
import os
import json
import argparse
from sklearn.metrics import accuracy_score
from data.dataloader import create_data_loaders
from models.model import DynamicGCN

os.makedirs("outputs", exist_ok=True)

def run_trainer(epoch, features_file, labels_file, best_params_file="outputs/best_params.json"):
    # Load best hyperparameters from file
    with open(best_params_file, "r") as f:
        best_params = json.load(f)
    
    # Extracting model parameters from the best_params JSON file
    num_closest = best_params['num_closest']  # Extract num_closest from the best_params file
    train_loader, val_loader, test_loader, label_encoder, features, labels = create_data_loaders(
        num_closest=num_closest,
        features_file=features_file,
        labels_file=labels_file
    )    
    # Setup model architecture using the hyperparameters
    hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(best_params['num_layers'])]
    layer_types = [GCNConv if best_params[f'layer_type_{i}'] == 'GCNConv' else SAGEConv for i in range(best_params['num_layers'])]
    
    final_model = DynamicGCN(input_dim=features.size(1), hidden_dims=hidden_dims, output_dim=len(label_encoder.classes_), layer_types=layer_types)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    # Train the model
    final_model.train()
    best_model_state = None
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_epoch = -1
    
    for epoch in range(epoch):  # Using the passed `epoch` argument here
        total_train_loss = 0
        for data in train_loader:
            final_optimizer.zero_grad()
            out = final_model(data.x, data.edge_index, data.edge_attr)
            loss = nn.CrossEntropyLoss()(out, data.y)
            loss.backward()
            final_optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss / len(train_loader))
    
        final_model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data in val_loader:
                out = final_model(data.x, data.edge_index, data.edge_attr)
                preds = out.argmax(dim=1)
                correct_val += (preds == data.y).sum().item()
                total_val += data.y.size(0)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)
    
        print(f'Epoch {epoch + 1}, Loss: {train_losses[-1]}, Val Accuracy: {val_accuracy:.4f}')
    
        # Check for best validation accuracy and update the model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = final_model.state_dict()
            best_epoch = epoch + 1
        final_model.train()
    
    print(f'Best epoch selected for the test set: {best_epoch}')
    final_model.load_state_dict(best_model_state)
    torch.save(final_model.state_dict(), "outputs/best_model.pth")
    
    # Plot training curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/final_loss_accuracy.png', dpi=300)
    plt.close()
    
    return final_model, test_loader, label_encoder


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Train a GNN model")
    parser.add_argument('--epoch', type=int, default=2000, help="Number of epochs to train the model")
    parser.add_argument('--f1', type=str, required=True, help="Path to the features file")
    parser.add_argument('--f2', type=str, required=True, help="Path to the labels file")
    args = parser.parse_args()

    # Call the function with the arguments
    run_trainer(epoch=args.epoch, features_file=args.f1, labels_file=args.f2)
