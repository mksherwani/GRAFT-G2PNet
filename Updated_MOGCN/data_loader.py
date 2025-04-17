import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_split_data(omic1_path='data/fpkm_data.csv', 
                        omic2_path='data/rppa_data.csv', 
                        label_path='data/sample_classes.csv', 
                        test_size=0.2, 
                        random_state=42,
                        n_features=1000):
    """Load and preprocess data with robust label handling"""
    # Load omics data
    omic1 = pd.read_csv(omic1_path, index_col=0)
    omic2 = pd.read_csv(omic2_path, index_col=0)
    labels = pd.read_csv(label_path, index_col=0)

    # Feature selection
    omic1 = omic1[omic1.var().sort_values(ascending=False).head(n_features).index]
    omic2 = omic2[omic2.var().sort_values(ascending=False).head(n_features).index]

    # Robust label processing
    if len(labels.columns) > 1:
        # If multiple columns, use first column that looks like labels
        for col in labels.columns:
            if labels[col].nunique() < 10:  # Heuristic for label column
                labels = labels[[col]]
                break
        else:
            raise ValueError("Could not identify label column")
    
    # Convert labels to numeric if needed
    if not pd.api.types.is_numeric_dtype(labels.squeeze()):
        le = LabelEncoder()
        labels = pd.Series(le.fit_transform(labels.squeeze()), index=labels.index)
    else:
        labels = labels.squeeze()
    
    # Ensure labels are properly 0-indexed
    labels = labels - labels.min()  # Shift to start at 0
    print(f"Final label range: {labels.min()} to {labels.max()}")

    # Align samples
    common_samples = omic1.index.intersection(omic2.index).intersection(labels.index)
    omic1 = omic1.loc[common_samples]
    omic2 = omic2.loc[common_samples]
    labels = labels.loc[common_samples]

    # Split data
    train_idx, test_idx = train_test_split(
        common_samples,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    return (
        omic1.loc[train_idx], omic1.loc[test_idx],
        omic2.loc[train_idx], omic2.loc[test_idx],
        labels.loc[train_idx], labels.loc[test_idx],
        train_idx, test_idx
    )
