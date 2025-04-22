import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def load_data(features_file, labels_file):
    features_df = pd.read_csv(features_file, delimiter=',')
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df.iloc[:, 1:])
    features = torch.tensor(scaled_features, dtype=torch.float)
    
    labels_df = pd.read_csv(labels_file, delimiter=',')
    label_encoder = LabelEncoder()
    labels_numerical = label_encoder.fit_transform(labels_df['Size'])
    labels = torch.tensor(labels_numerical, dtype=torch.long)    
    return features_df, features, labels, label_encoder


def calculate_closest_distances(genes_data, num_closest=8):
    # genes_data is a DataFrame containing the scaled features
    gene_distances = pdist(genes_data, metric='euclidean')
    gene_distance_matrix = squareform(gene_distances)

    min_nonzero_distance = np.min(gene_distances[gene_distances > 0]) if np.any(gene_distances > 0) else 1e-5
    gene_distance_matrix[gene_distance_matrix == 0] = min_nonzero_distance

    results = []
    added_pairs = set()

    num_genes = gene_distance_matrix.shape[0]
    for i in range(num_genes):
        sorted_distances = sorted(
            [(gene_distance_matrix[i, j], j) for j in range(num_genes) if i != j]
        )
        closest_distances = sorted_distances[:num_closest]
        for distance, j in closest_distances:
            pair = tuple(sorted([i, j]))
            if pair in added_pairs:
                continue
            added_pairs.add(pair)
            results.append([i, j, distance])

    result_df = pd.DataFrame(results, columns=['Gene1', 'Gene2', 'RawDistance'])
    min_distance = result_df['RawDistance'].min()
    max_distance = result_df['RawDistance'].max()
    result_df['NormalizedDistance'] = 0.001 + 0.999 * ((result_df['RawDistance'] - min_distance) / (max_distance - min_distance))
    return result_df[['Gene1', 'Gene2', 'NormalizedDistance']]


def create_edge_index(edge_index, edge_weights, node_count):
    mask = (edge_index[0] < node_count) & (edge_index[1] < node_count)
    return edge_index[:, mask], edge_weights[mask]

def create_data_loaders(num_closest, features_file, labels_file):
    features_df, features, labels, label_encoder = load_data(features_file, labels_file)
    
    # Split the data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=1)
    
    # Calculate gene distance matrix based on scaled features (using columns only from features_df)
    adjacency_df = calculate_closest_distances(features_df.iloc[:, 1:], num_closest=num_closest)
    gene1_indices = adjacency_df['Gene1'].values
    gene2_indices = adjacency_df['Gene2'].values
    edges = torch.tensor(np.array([gene1_indices, gene2_indices]), dtype=torch.long)
    edge_weights = torch.tensor(adjacency_df['NormalizedDistance'].values, dtype=torch.float)
    
    # Create edge indices for each subset
    edge_index_train, edge_weights_train = create_edge_index(edges, edge_weights, X_train.size(0))
    edge_index_val, edge_weights_val = create_edge_index(edges, edge_weights, X_val.size(0))
    edge_index_test, edge_weights_test = create_edge_index(edges, edge_weights, X_test.size(0))
    
    train_data = Data(x=X_train, edge_index=edge_index_train, edge_attr=edge_weights_train, y=y_train)
    val_data = Data(x=X_val, edge_index=edge_index_val, edge_attr=edge_weights_val, y=y_val)
    test_data = Data(x=X_test, edge_index=edge_index_test, edge_attr=edge_weights_test, y=y_test)
    
    train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=32, shuffle=False)
    test_loader = DataLoader([test_data], batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, label_encoder, features, labels
