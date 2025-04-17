# extract_weights.py
import torch
import numpy as np
import pandas as pd

def extract_feature_importance(model_path, omic1_features, omic2_features, omic1_std, omic2_std, topn=100):
    model = torch.load(model_path, map_location='cpu')
    state = model.state_dict()
    # Get the weight matrices from the first linear layer of each encoder
    # For omics1, assuming encoder_omics1[0] is nn.Linear(in_dim1, 64)
    weight_omic1 = state['encoder_omics1.0.weight'].detach().cpu().numpy()  # shape (64, in_dim1)
    weight_omic2 = state['encoder_omics2.0.weight'].detach().cpu().numpy()  # shape (64, in_dim2)
    
    # Compute absolute weight sums per feature (i.e., sum the absolute values along the output dimension)
    importance_omic1 = np.sum(np.abs(weight_omic1), axis=0) * omic1_std.values
    importance_omic2 = np.sum(np.abs(weight_omic2), axis=0) * omic2_std.values
    
    # Form DataFrames
    df_omic1 = pd.DataFrame({
        'Feature': omic1_features,
        'Importance': importance_omic1
    })
    df_omic2 = pd.DataFrame({
        'Feature': omic2_features,
        'Importance': importance_omic2
    })
    # Select top N features for each omic
    top_omic1 = df_omic1.nlargest(topn, 'Importance')
    top_omic2 = df_omic2.nlargest(topn, 'Importance')
    
    return top_omic1, top_omic2

# Usage example:
# top_features_omic1, top_features_omic2 = extract_feature_importance('model/MMAE2Omics_epoch_10.pkl', 
#     omic1_features=list(omic1_train_df.columns),
#     omic2_features=list(omic2_train_df.columns),
#     omic1_std=omic1_train_df.std(),
#     omic2_std=omic2_train_df.std(),
#     topn=100)
# top_features_omic1.to_csv('result/topn_omic1.csv', index=False)
# top_features_omic2.to_csv('result/topn_omic2.csv', index=False)
