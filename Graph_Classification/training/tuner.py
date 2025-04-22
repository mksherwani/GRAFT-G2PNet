import optuna
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
import matplotlib.pyplot as plt
import os
import json
import torch

from data.dataloader import create_data_loaders
from models.model import DynamicGCN

os.makedirs("outputs", exist_ok=True)

def objective(trial, features_file, labels_file, epoch):
    # Optimizing number of layers, their types, and other parameters
    num_layers = trial.suggest_int('num_layers', 2, 3)
    hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 32, 128) for i in range(num_layers)]
    layer_types = [trial.suggest_categorical(f'layer_type_{i}', ['GCNConv', 'SAGEConv']) for i in range(num_layers)]
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    num_closest = trial.suggest_int('num_closest', 3, 10)

    # Get data loaders with the current num_closest parameter
    # Get data loaders with the current num_closest parameter
    train_loader, val_loader, test_loader, label_encoder, features, labels = create_data_loaders(
        num_closest=num_closest,
        features_file=features_file,
        labels_file=labels_file
    )

    # Initialize model with dynamic layer types and number of layers
    layer_types_objects = [GCNConv if layer_type == 'GCNConv' else SAGEConv for layer_type in layer_types]
    model = DynamicGCN(input_dim=features.size(1), hidden_dims=hidden_dims, output_dim=len(label_encoder.classes_), layer_types=layer_types_objects)
    criterion = nn.CrossEntropyLoss()
    optimizer = {
        'adam': torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        'adamw': torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay),
        'sgd': torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    }[optimizer_name]

    patience = 20
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epoch):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            preds = out.argmax(dim=1)
            correct_train += (preds == data.y).sum().item()
            total_train += data.y.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data in val_loader:
                out = model(data.x, data.edge_index, data.edge_attr)
                val_loss = criterion(out, data.y)
                total_val_loss += val_loss.item()
                preds = out.argmax(dim=1)
                correct_val += (preds == data.y).sum().item()
                total_val += data.y.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        trial.report(val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            raise optuna.TrialPruned()

    # Plot training and validation curves
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))
    axs[0].plot(train_losses, label='Training Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].plot(train_accuracies, label='Training Accuracy', color='red')
    axs[1].plot(val_accuracies, label='Validation Accuracy', color='teal')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(f'outputs/optuna_trial_{trial.number}_training_curves.png', dpi=300)
    plt.close()

    return best_val_loss

def save_trial_info_callback(study, trial):
    trial_info = f"Trial {trial.number}:\n"
    trial_info += f"  Value: {trial.value}\n"
    trial_info += "  Params:\n"
    for key, value in trial.params.items():
        trial_info += f"    {key}: {value}\n"
    with open(f"outputs/trial_{trial.number}.txt", "w") as f:
        f.write(trial_info)


def run_tuner(epoch, features_file, labels_file):
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=50, interval_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    
    # Pass features_file and labels_file along with the trial to the objective function
    study.optimize(
        lambda trial: objective(trial, features_file, labels_file, epoch),
        n_trials=500,
        callbacks=[save_trial_info_callback]
    )
        
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # Save best hyperparameters for later use
    with open("outputs/best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)

    return trial

