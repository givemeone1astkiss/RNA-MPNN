from torch import nn
import torch
from torch.nn import functional as F
from ..config.glob import NUM_MAIN_SEQ_ATOMS


class AtomPool(nn.Module):
    def __init__(self, raw_dim: int, atom_pool_hidden_dim: int, num_layers: int, dropout: float):
        """
        Args:
            raw_dim (int): Dimension of the raw residue-level features.
            atom_pool_hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of linear layers for computing weights.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.num_layers = num_layers
        self.atom_pool_hidden_dim = atom_pool_hidden_dim

        layers = []
        input_dim = raw_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, atom_pool_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = atom_pool_hidden_dim
        layers.append(nn.Linear(atom_pool_hidden_dim, NUM_MAIN_SEQ_ATOMS))
        self.weight_layers = nn.Sequential(*layers)

    def forward(self, atom_embedding: torch.Tensor, atom_mask: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """
        Pool atom-level features into residue-level features.

        Args:
            atom_embedding (torch.Tensor): Atom-level features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
            atom_mask (torch.Tensor): Atom mask of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            raw (torch.Tensor): Residue-level raw features of shape (batch_size, max_len, raw_dim).

        Returns:
            atom_embedding (torch.Tensor): Residue-level pooled features of shape (batch_size, max_len, atom_embedding_dim).
        """
        batch_size, max_len, _ = raw.shape

        atom_mask = atom_mask.reshape(batch_size, max_len, NUM_MAIN_SEQ_ATOMS)
        atom_embedding = atom_embedding.reshape(batch_size, max_len, NUM_MAIN_SEQ_ATOMS, -1)

        weights = self.weight_layers(raw)  # Shape: (batch_size, max_len, NUM_MAIN_SEQ_ATOMS)

        weights = weights * atom_mask
        weights = F.softmax(weights, dim=-1)
        pooled_atom_embedding = torch.sum(atom_embedding * weights.unsqueeze(-1), dim=2)
        return pooled_atom_embedding


class ResPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, res_embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass