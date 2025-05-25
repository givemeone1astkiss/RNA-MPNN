from torch import nn
from .functional import scatter_sum
import torch

class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_message_layers, num_dense_layers, dim_dense_layers, dropout=0.1, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        layers = []
        input_dim = num_hidden + num_in
        for i in range(num_message_layers):
            layers.append(nn.Linear(input_dim, num_hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = num_hidden
        self.message_layers = nn.Sequential(*layers)

        layers = []
        for i in range(num_dense_layers):
            layers.append(nn.Linear(input_dim, dim_dense_layers))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = dim_dense_layers
        layers.append(nn.Linear(input_dim, num_hidden))
        self.dense = nn.Sequential(*layers)

    def forward(self, h_V, h_E, edge_idx)->torch.Tensor:
        src_idx, _ = edge_idx[0], edge_idx[1]
        h_message = self.message_layers(h_E)
        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + dh)
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + dh)
        return h_V