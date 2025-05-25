import torch
import torch.nn as nn
from typing import Optional
from ..utils.data import separate, concat, gen_mask
import math
from ..config.glob import SEPS

class GraphNormalization(nn.Module):
    def __init__(self, embedding_dim: int):
        """
        Graph normalization layer for normalizing node features in a graph.
        Args:
            embedding_dim (int): Dimension of the node features.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, embedding_dim))
        self.shift = nn.Parameter(torch.zeros(1, 1, embedding_dim))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Normalize graph node features.

        Args:
            features (torch.Tensor): Input node features of shape (batch_size, num_nodes, embedding_dim).
            mask (torch.Tensor): Mask indicating valid nodes of shape (batch_size, num_nodes).

        Returns:
            torch.Tensor: Normalized node features of shape (batch_size, num_nodes, embedding_dim).
        """
        # Expand mask to match feature dimensions
        mask = mask.unsqueeze(-1)  # Shape: (batch_size, num_nodes, 1)

        # Compute mean and variance, excluding padding nodes
        masked_features = features * mask
        valid_counts = mask.sum(dim=1, keepdim=True)
        valid_counts[valid_counts == 0] = 1  # Avoid division by zero

        mean = masked_features.sum(dim=1, keepdim=True) / valid_counts
        variance = ((masked_features - mean) ** 2).sum(dim=1, keepdim=True) / valid_counts
        std = torch.sqrt(variance + SEPS)

        # Normalize features
        normalized_features = (features - mean) / std
        normalized_features = normalized_features * self.scale + self.shift

        # Ensure padding atoms have zero features
        normalized_features = normalized_features * mask

        return normalized_features

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias

class Readout(nn.Module):
    def __init__(self, embedding_dim: int, readout_hidden_dim: int, num_layers: int, dropout: float = 0.1):
        """
        Initialize the Readout module for residue-level classification.
        Args:
            embedding_dim: The dimension of the residue embedding.
            readout_hidden_dim: The dimension of the hidden layers in the readout network.
            num_layers: The number of layers in the readout network.
            dropout: The dropout rate for regularization.
        """
        super().__init__()
        layers = []
        input_dim = embedding_dim

        # Build feedforward layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, readout_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = readout_hidden_dim

        # Final layer for classification
        layers.append(nn.Linear(input_dim, 4))
        self.readout_layers = nn.Sequential(*layers)

    def forward(self, res_embedding: torch.Tensor) -> torch.Tensor:

        logits = self.readout_layers(res_embedding)

        return logits

class RNABert(nn.Module):
    def __init__(self,
                 padding_len: int,
                 hidden_dim: int,
                 num_attn_layers: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_ffn_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        self.padding_len = padding_len

        layers = []
        for _ in range(num_attn_layers):
            layers.append(nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True))

        self.bi_attention_layers = nn.ModuleList(layers)

        layers = []
        for _ in range(num_attn_layers):
            layers.append(GraphNormalization(hidden_dim))
        self.graph_norm_layers = nn.ModuleList(layers)

        layers = []
        input_dim = hidden_dim
        for _ in range(num_ffn_layers):
            layers.append(nn.Linear(input_dim, ffn_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = ffn_dim
        layers.append(nn.Linear(ffn_dim, hidden_dim))
        self.ffn_layers = nn.Sequential(*layers)

    @staticmethod
    def _position_embedding(padded_h_V: torch.Tensor, padded_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            padded_h_V (torch.Tensor): Residue-level features of shape (batch_size, padding_len, embedding_dim).
            padded_mask (torch.Tensor): Padded mask indicating valid residues of shape (batch_size, padding_len).

        Returns:
            padded_h_V (torch.Tensor): Residue-level features with position embeddings added.
        """
        _, padding_len, embedding_dim = padded_h_V.size()
        position = torch.arange(0, padding_len, device=padded_h_V.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=padded_h_V.device) * -(
                    math.log(10000.0) / embedding_dim))

        pe = torch.zeros(padding_len, embedding_dim, device=padded_h_V.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        padded_h_V = padded_h_V + pe.unsqueeze(0)
        padded_h_V = padded_h_V * padded_mask.unsqueeze(-1)

        return padded_h_V

    def _padding(self, res_embedding: torch.Tensor, mask: torch.Tensor):
        padded_h_V = torch.cat((res_embedding,
                                          torch.zeros(res_embedding.shape[0], self.padding_len - res_embedding.shape[1],
                                                      res_embedding.shape[2]).to(res_embedding.device)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros(mask.shape[0], self.padding_len - mask.shape[1]).to(mask.device)),
                                dim=1)
        return padded_h_V, padded_mask

    def forward(self, h_V: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sep_h_V = separate(h_V, lengths)
        padded_h_V, padded_mask = self._padding(sep_h_V, gen_mask(lengths).to(sep_h_V.device))
        attention_mask = ~padded_mask.bool()
        for attention_layer, norm_layer in zip(self.bi_attention_layers, self.graph_norm_layers):
            padded_h_V += attention_layer(query=padded_h_V,
                                            key=padded_h_V,
                                            value=padded_h_V,
                                            key_padding_mask=attention_mask.to(padded_h_V.device))[0]
            padded_h_V = norm_layer(padded_h_V, padded_mask)
        padded_h_V = self.ffn_layers(padded_h_V)
        padded_h_V *= padded_mask.unsqueeze(-1)
        return concat(padded_h_V[:, :sep_h_V.shape[1], :], lengths)