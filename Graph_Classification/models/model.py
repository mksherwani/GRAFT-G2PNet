import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, layer_types):
        super(DynamicGCN, self).__init__()
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(layer_types[0](input_dim, hidden_dims[0]))
        # Additional layers
        for i in range(1, len(hidden_dims)):
            self.convs.append(layer_types[i](hidden_dims[i-1], hidden_dims[i]))
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.fc(x), dim=1)
