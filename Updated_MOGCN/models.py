import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MoGCN(nn.Module):
    def __init__(self, input_dims, hidden_dim=64, num_classes=4, dropout=0.5):
        super(MoGCN, self).__init__()
        # First omic (omic1) GCN layers
        self.gc1_omic1 = GCNConv(input_dims[0], hidden_dim)
        self.gc2_omic1 = GCNConv(hidden_dim, hidden_dim)

        # Second omic (omic2) GCN layers
        self.gc1_omic2 = GCNConv(input_dims[1], hidden_dim)
        self.gc2_omic2 = GCNConv(hidden_dim, hidden_dim)

        # Dropouts
        self.dp = nn.Dropout(dropout)

        # Final classifier
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x_omic1, x_omic2, edge_index):
        # omic1 branch
        x1 = F.elu(self.gc1_omic1(x_omic1, edge_index))
        x1 = self.dp(x1)
        x1 = F.elu(self.gc2_omic1(x1, edge_index))
        x1 = self.dp(x1)

        # omic2 branch
        x2 = F.elu(self.gc1_omic2(x_omic2, edge_index))
        x2 = self.dp(x2)
        x2 = F.elu(self.gc2_omic2(x2, edge_index))
        x2 = self.dp(x2)

        # Concatenate both feature vectors
        x = torch.cat([x1, x2], dim=1)

        # Final prediction
        out = self.fc(x)
        return out
