# GRAFT-G2PNet
GRAFT-G2PNet: Graph Based Adaptive Feature tuning, Genotype2Phenotype network

# 🧠 Graph Neural Network Classifier with Optuna Tuning & Feature Selection

A modular, high-performance **Graph Neural Network (GNN)** classifier using **PyTorch Geometric**, with:

- 🧬 **Autoencoder-based Feature Selection**
- 🔍 **Optuna Hyperparameter Optimization**
- 🧠 **Dynamic GNN Architectures**
- 📊 **Visualization & Evaluation**
- 🛠️ Command-line control for full pipeline

---

## 📁 Project Structure

```
.
├──Graph_Classification
    ├── config
          ├── config.yaml       # Configuration file
    ├── data
          ├── dataloader.py       # Loads features, labels, creates edge indices
    ├── models           
          ├── model.py           # Autoencoder and dynamic GNN models
    ├── training           
          ├── trainer.py          # Training loop and early stopping
          ├── tuner.py            # Optuna-based hyperparameter tuning
          ├── evaluator.py        # Testing + confusion matrix visualization
    ├── utils
          ├── utils.py            # Helper functions (AE training, edge gen)
    ├── main.py             # Entry point: training / tuning / evaluation
├──Updated_MOGCN
└── README.md           # You're here!
```

---

## ⚙️ Installation

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install optuna scikit-learn matplotlib pandas
```

---

## 🧪 Pipeline Overview

### 1. 🔬 Feature Selection with Autoencoder

High-dimensional genotype features are compressed into a lower-dimensional representation using an Autoencoder.

```python
AutoEncoder(input_dim, hidden_dim, compressed_dim)
```

➡️ Output: compressed features for GNN input.

---

### 2. 🔗 Edge Creation

Edges are formed between genotypes using **Euclidean distance** in feature space.

```python
calculate_closest_distances(df, num_closest=8)
```

➡️ Keeps only `k` closest connections per node to build sparse graph.

---

### 3. 🧠 Dynamic GNN Model

Builds a **flexible GNN** using a sequence of:

- `GCNConv` or `SAGEConv` (chosen per layer)
- Batch normalization
- Dropout + ReLU
- Global mean pooling
- Final classifier layer

```python
DynamicGCN(input_dim, hidden_dims, output_dim, layer_types)
```

---

### 4. 🔍 Hyperparameter Tuning with Optuna

Optuna searches for the best combination of:

| Hyperparameter | Range / Options           |
|----------------|---------------------------|
| Num Layers     | 2 to 3                    |
| Layer Types    | GCNConv / SAGEConv        |
| Hidden Sizes   | 32 to 96 per layer        |
| Optimizer      | Adam / AdamW / SGD        |
| Learning Rate  | 1e-4 to 1e-1 (log scale)  |
| Weight Decay   | 1e-5 to 1e-1 (log scale)  |
| Edge K         | 3 to 10 (neighbors)       |

Results are logged and visualized per trial.

---

### 5. 🏋️ Training

- Cross-entropy loss
- Accuracy tracking
- Early stopping with patience
- Saves loss/accuracy curves per trial

```bash
python main.py --mode train --f1 <features.txt> --f2 <labels.txt> --epoch 200
```

---

### 6. 📈 Evaluation

- Loads best model
- Evaluates on test set
- Plots confusion matrix and final loss/accuracy curves

```bash
python main.py --mode eval --f1 <features.txt> --f2 <labels.txt>
```

---

## 🚀 Command Line Usage

### 🔧 Tune with Optuna

```bash
python main.py --mode tune --f1 path/to/input.txt --f2 path/to/labels.txt --epoch 50
```

### 🏋️ Train Final Model

```bash
python main.py --mode train --f1 path/to/input.txt --f2 path/to/labels.txt --epoch 200
```

### 📊 Evaluate Model

```bash
python main.py --mode eval --f1 path/to/input.txt --f2 path/to/labels.txt
```

---

## 📊 Output Visuals

All visualizations are saved in the `outputs/` folder:

- `optuna_trial_X_training_curves.png`
- `final_loss_accuracy.png`
- `final_confusion_matrix.png`

---

## 💡 Key Features

- ✅ Autoencoder compresses noisy input features
- ✅ Edges dynamically built from input data
- ✅ GNN layer types selected **per layer**
- ✅ All training / tuning / eval via single CLI
- ✅ Easy to adapt for other graph datasets

---

## 📚 Dependencies

- `torch`, `torch-geometric`
- `optuna`
- `scikit-learn`
- `matplotlib`, `pandas`

---

## 🙌 Credits

Built with ❤️ using:

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Optuna](https://optuna.org/)
- [Scikit-learn](https://scikit-learn.org/)
