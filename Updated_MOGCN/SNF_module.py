# snf_module.py
import snf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def build_snf_network(omics_list, metric='sqeuclidean', K=20, mu=0.5):
    """
    omics_list: list of latent representations (each as a numpy array)
    """
    # Compute affinity networks using the snf package
    affinity_networks = snf.make_affinity(omics_list, metric=metric, K=K, mu=mu)
    # Fuse the networks
    fused_network = snf.snf(affinity_networks, K=K)
    
    return fused_network

def visualize_snf(fused_network, sample_ids, output_prefix='SNF'):
    # Convert to DataFrame
    fused_df = pd.DataFrame(fused_network, index=sample_ids, columns=sample_ids)
    fused_df.to_csv(f'result/{output_prefix}_fused_matrix.csv')
    
    # Remove self-connections for visualization
    np.fill_diagonal(fused_df.values, 0)
    
    # Plot and save a clustermap using seaborn
    clustermap = sns.clustermap(fused_df, cmap='vlag', figsize=(8,8))
    clustermap.savefig(f'result/{output_prefix}_fused_clustermap.png', dpi=300)
