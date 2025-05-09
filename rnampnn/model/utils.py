import torch
from torch import nn
from ..config.glob import SEPS, NUM_RES_TYPES
import math

class GraphNormalization(nn.Module):
    def __init__(self, embedding_dim: int):
        """
        Graph normalization layer for normalizing node features in a graph.
        Args:
            embedding_dim (int): Dimension of the node features.
            epsilon (float): Small constant to avoid division by zero.
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


class Readout(nn.Module):
    def __init__(self, res_embedding_dim: int, readout_hidden_dim: int, num_layers: int, dropout: float = 0.1):
        """
        Initialize the Readout module for residue-level classification.
        Args:
            res_embedding_dim: The dimension of the residue embedding.
            readout_hidden_dim: The dimension of the hidden layers in the readout network.
            num_layers: The number of layers in the readout network.
            dropout: The dropout rate for regularization.
        """
        super().__init__()
        layers = []
        input_dim = res_embedding_dim

        # Build feedforward layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, readout_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = readout_hidden_dim

        # Final layer for classification
        layers.append(nn.Linear(input_dim, NUM_RES_TYPES))
        self.readout_layers = nn.Sequential(*layers)

    def forward(self, res_embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            res_embedding (torch.Tensor): Node features of shape (batch_size, max_len, res_embedding_dim).
            mask (torch.Tensor): Mask indicating valid residues of shape (batch_size, max_len).

        Returns:
            logits (torch.Tensor): Predicted residue type logits of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS).
        """

        logits = self.readout_layers(res_embedding)  # Shape: (batch_size, max_len, NUM_MAIN_SEQ_ATOMS)

        logits *= mask.unsqueeze(-1)  # Set predictions for padding residues to zero

        return logits


class BertReadout(nn.Module):
    def __init__(self,
                 padding_len: int,
                 res_embedding_dim: int,
                 num_attn_layers: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_ffn_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        self.padding_len = padding_len

        layers = []
        for _ in range(num_attn_layers-1):
            layers.append(nn.MultiheadAttention(embed_dim=res_embedding_dim,
                                                  num_heads=num_heads,
                                                  dropout=dropout,
                                                  batch_first=True))

        self.bi_attention_layers = nn.Sequential(*layers)
        self.graph_norm = GraphNormalization(res_embedding_dim)
        layers = []
        input_dim = res_embedding_dim
        for _ in range(num_ffn_layers-1):
            layers.append(nn.Linear(input_dim, ffn_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = ffn_dim
        layers.append(nn.Linear(input_dim, NUM_RES_TYPES))
        self.ffn_layers = nn.Sequential(*layers)

    def _padding(self, res_embedding: torch.Tensor, mask: torch.Tensor):
        padded_res_embedding = torch.cat((res_embedding, torch.zeros(res_embedding.shape[0], self.padding_len - res_embedding.shape[1], res_embedding.shape[2])), dim=1)
        padded_mask = torch.cat((mask, torch.zeros(mask.shape[0], self.padding_len - mask.shape[1])), dim=1)
        return padded_res_embedding, padded_mask

    @staticmethod
    def _position_embedding(padded_res_embedding: torch.Tensor, padded_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            padded_res_embedding (torch.Tensor): Residue-level features of shape (batch_size, padding_len, embedding_dim).
            padded_mask (torch.Tensor): Padded mask indicating valid residues of shape (batch_size, padding_len).

        Returns:
            padded_res_embedding (torch.Tensor): Residue-level features with position embeddings added.
        """
        batch_size, padding_len, embedding_dim = padded_res_embedding.size()
        position = torch.arange(0, padding_len, device=padded_res_embedding.device).unsqueeze(1)  # Shape: (padding_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=padded_res_embedding.device) * -(math.log(10000.0) / embedding_dim))  # Shape: (embedding_dim // 2)

        pe = torch.zeros(padding_len, embedding_dim, device=padded_res_embedding.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        padded_res_embedding = padded_res_embedding + pe.unsqueeze(0)
        padded_res_embedding = padded_res_embedding * padded_mask.unsqueeze(-1)

        return padded_res_embedding

    def forward(self, res_embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padded_res_embedding, padded_mask = self._padding(res_embedding, mask)
        attention_mask = ~padded_mask.bool()

        for layer in self.bi_attention_layers:
            padded_res_embedding += layer(query=self.graph_norm(padded_res_embedding, padded_mask),
                                            key=self.graph_norm(padded_res_embedding, padded_mask),
                                            value=self.graph_norm(padded_res_embedding, padded_mask),
                                            key_padding_mask=attention_mask)[0]

        logits = self.ffn_layers(padded_res_embedding)
        logits *= padded_mask.unsqueeze(-1)
        return logits[:, :res_embedding.shape[1], :]


