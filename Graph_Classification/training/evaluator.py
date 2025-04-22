import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import json
from data.dataloader import create_data_loaders
from models.model import DynamicGCN
from torch_geometric.nn import GCNConv, SAGEConv

def run_evaluator(features_file, labels_file):
    with open("outputs/best_params.json", "r") as f:
        best_params = json.load(f)
    num_closest = best_params["num_closest"]
    
    train_loader, val_loader, test_loader, label_encoder, features, labels = create_data_loaders(
        features_file=features_file,
        labels_file=labels_file,
        num_closest=num_closest
    )

    hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(best_params['num_layers'])]
    layer_types = [GCNConv if best_params[f'layer_type_{i}'] == 'GCNConv' else SAGEConv for i in range(best_params['num_layers'])]
    
    final_model = DynamicGCN(input_dim=features.size(1), hidden_dims=hidden_dims, output_dim=len(label_encoder.classes_), layer_types=layer_types)
    final_model.load_state_dict(torch.load("outputs/best_model.pth"))
    final_model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            out = final_model(data.x, data.edge_index, data.edge_attr)
            preds = out.argmax(dim=1)
            test_preds.append(preds)
            test_labels.append(data.y)
    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)
    
    acc = accuracy_score(test_labels.numpy(), test_preds.numpy())
    class_report = classification_report(test_labels.numpy(), test_preds.numpy(), target_names=label_encoder.classes_)
    
    # Save accuracy and classification report to text file
    with open("outputs/final_results.txt", "w") as result_file:
        result_file.write(f"Test Accuracy: {acc:.4f}\n\n")
        result_file.write("Classification Report:\n")
        result_file.write(class_report)

    # Confusion matrix
    confusion_mtx = confusion_matrix(test_labels.numpy(), test_preds.numpy())
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    thresh = confusion_mtx.max() / 2.0
    for i, j in np.ndindex(confusion_mtx.shape):
        plt.text(j, i, f'{confusion_mtx[i, j]}', ha='center', va='center', color='white' if confusion_mtx[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.savefig('outputs/final_confusion_matrix.png', dpi=300)
    plt.close()
    
if __name__ == '__main__':
    run_evaluator("your_features_file.csv", "your_labels_file.csv")
