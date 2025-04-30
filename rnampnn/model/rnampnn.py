import torch
import torch.nn as nn
from networkx.classes.filters import hide_edges
from pytorch_lightning import LightningModule
from typing import Tuple
from torch import Tensor
from ..config.glob import NUM_MAIN_SEQ_ATOMS
from torch.nn import functional as F

class GraphNormalization(nn.Module):
    def __init__(self, embedding_dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
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
        std = torch.sqrt(variance + self.epsilon)

        # Normalize features
        normalized_features = (features - mean) / std
        normalized_features = normalized_features * self.scale + self.shift

        # Ensure padding atoms have zero features
        normalized_features = normalized_features * mask

        return normalized_features


class AtomFeature(nn.Module):
    def __init__(self, num_neighbour: int, embedding_dim: int = 32):
        super().__init__()
        self.num_neighbour = num_neighbour
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=NUM_MAIN_SEQ_ATOMS, embedding_dim=self.embedding_dim)
        self.normalization = GraphNormalization(embedding_dim=self.embedding_dim)

    def _get_atom_graph(self, atom_coords: torch.Tensor, atom_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a k-nearest neighbor graph for RNA atom coordinates.

        Args:
            atom_coords (torch.Tensor): Atom coordinates of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, 3).
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - dist_neighbors: Distances to neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).
                - edge_index: Indices of neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).
        """
        batch_size, num_atoms, _ = atom_coords.shape

        # Expand mask for pairwise distance calculation
        atom_mask_2d = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)  # (batch_size, num_atoms, num_atoms)

        # Compute pairwise distances
        d_coords = atom_coords.unsqueeze(1) - atom_coords.unsqueeze(2)  # (batch_size, num_atoms, num_atoms, 3)
        distances = torch.sqrt(torch.sum(d_coords ** 2, dim=-1) + 1e-6)  # (batch_size, num_atoms, num_atoms)

        # Mask invalid distances (real atoms to padding atoms)
        distances = distances * atom_mask_2d + (1.0 - atom_mask_2d) * 1e6  # Large value for invalid distances

        # Select k-nearest neighbors
        dist_neighbors, edge_idx = torch.topk(distances, min(self.num_neighbour, num_atoms), dim=-1, largest=False)

        # Handle cases where num_neighbour exceeds the number of valid atoms
        if self.num_neighbour > num_atoms:
            padding_size = self.num_neighbour - num_atoms
            dist_neighbors = torch.cat(
                [dist_neighbors, torch.full((batch_size, num_atoms, padding_size), 1e6, device=distances.device)],
                dim=-1
            )
            edge_idx = torch.cat(
                [edge_idx,
                 torch.full((batch_size, num_atoms, padding_size), -1, device=distances.device, dtype=torch.long)],
                dim=-1
            )

        # Set all neighbors of padding atoms to -1 and distances to 1e6
        padding_atom_mask = (atom_mask == 0).unsqueeze(-1)  # (batch_size, num_atoms, 1)
        edge_idx = edge_idx.masked_fill(padding_atom_mask, 0)
        dist_neighbors = dist_neighbors.masked_fill(padding_atom_mask, 1e6)

        return dist_neighbors, edge_idx

    def _encoder(self, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate embedding vectors for atom types based on the cyclic pattern.

        Args:
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            torch.Tensor: Embedding vectors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, embedding_dim).
        """
        batch_size, num_atoms = atom_mask.shape

        # Generate cyclic atom types (0, 1, 2, 3, 4, 5, 6) for valid atoms
        atom_types = torch.arange(num_atoms, device=atom_mask.device) % NUM_MAIN_SEQ_ATOMS
        atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, num_atoms)

        # Generate embeddings for atom types
        embeddings = self.embedding(atom_types)  # Shape: (batch_size, num_atoms, embedding_dim)

        # Set embeddings for padding atoms (where atom_mask == 0) to zero
        embeddings = embeddings * atom_mask.unsqueeze(-1)

        # Normalize embeddings
        return self.normalization(embeddings, atom_mask)

    @staticmethod
    def _to_atom_format(coords: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the coordinates and mask to atom format.
        Args:
            coords: Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask: Mask of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS).

        Returns:
            atom_coords: Coordinates reshaped to (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, 3).
            atom_mask: Mask reshaped to (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
        """
        atom_coords = coords.reshape(coords.shape[0], -1, coords.shape[3])
        print(mask.shape)
        atom_mask = mask.unsqueeze(-1).expand(-1, -1, NUM_MAIN_SEQ_ATOMS).reshape(mask.shape[0], -1)
        return atom_coords, atom_mask

    def forward(self, coords: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass to process the coordinates and mask.
        Args:
            coords: Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask: Mask of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS).

        Returns:
            atom_coords: Coordinates reshaped to (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, 3).
            atom_mask: Mask reshaped to (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            encode: Encoded atom features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, embedding_dim).
            dist_neighbors: Distances to neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).
            edge_index: Indices of neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).
        """
        atom_coords, atom_mask = self._to_atom_format(coords, mask)
        encode = self._encoder(atom_mask)
        dist_neighbors, edge_index = self._get_atom_graph(atom_coords, atom_mask)
        return atom_coords, atom_mask, encode, dist_neighbors, edge_index


class AtomMPNN(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.graph_norm = GraphNormalization(embedding_dim=hidden_dim)

        # Create a sequential module for message generation
        layers = []
        input_dim = hidden_dim * 2 + 1
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.message_layers = nn.Sequential(*layers)

    def message(self, atom_mask: torch.Tensor, encode: torch.Tensor, dist_neighbors: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Generate messages for each edge in the graph.

        Args:
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            encode (torch.Tensor): Node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, hidden_dim).
            dist_neighbors (torch.Tensor): Distances to neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).

        Returns:
            torch.Tensor: Messages of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour, hidden_dim).
        """
        # Mask invalid nodes and expand encode for neighbors
        encode = encode * atom_mask.unsqueeze(-1)
        source_features = torch.gather(
            encode.unsqueeze(2).expand(-1, -1, edge_index.size(-1), -1),
            1,
            edge_index.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        )

        # Concatenate source, target, and distance features
        edge_features = torch.cat([
            source_features,
            encode.unsqueeze(2).expand(-1, -1, edge_index.size(-1), -1),
            dist_neighbors.unsqueeze(-1)
        ], dim=-1)

        # Generate messages and mask invalid ones

        valid_mask = (edge_index != -1).unsqueeze(-1).float()
        messages = self.message_layers(edge_features) * valid_mask
        return messages

    @staticmethod
    def aggregation(atom_mask: torch.Tensor, encode: torch.Tensor, messages: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages for each node and update node features.

        Args:
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            encode (torch.Tensor): Node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, hidden_dim).
            messages (torch.Tensor): Messages of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour, hidden_dim).
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).

        Returns:
            torch.Tensor: Updated node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, hidden_dim).
        """
        # Compute the sum of messages for each target node
        message_sum = torch.sum(messages, dim=2)

        # Compute the number of valid neighbors for each node
        valid_neighbors = (edge_index != -1).sum(dim=-1, keepdim=True).float()
        valid_neighbors[valid_neighbors == 0] = 1  # Avoid division by zero

        # Compute the mean of messages
        aggregated_messages = message_sum / valid_neighbors

        # Add aggregated messages to the original node features
        updated_encode = encode + aggregated_messages

        # Mask invalid nodes (padding atoms)
        updated_encode = updated_encode * atom_mask.unsqueeze(-1)
        return updated_encode

    def forward(self, atom_encode: torch.Tensor, atom_mask: torch.Tensor, dist_neighbors: torch.Tensor, edge_index: torch.Tensor) -> \
    tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the AtomMPNN module.

        Args:
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            atom_encode (torch.Tensor): Node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, hidden_dim).
            dist_neighbors (torch.Tensor): Distances to neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - atom_encode: Updated node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, hidden_dim).
                - atom_mask: Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
                - dist_neighbors: Distances to neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).
                - edge_index: Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_neighbour).
        """
        messages = self.message(atom_mask, atom_encode, dist_neighbors, edge_index)
        atom_encode = self.aggregation(atom_mask, atom_encode, messages, edge_index)
        atom_encode = self.graph_norm(atom_encode, atom_mask)
        return atom_encode, atom_mask, dist_neighbors, edge_index

class ResFeature(nn.Module):
    def __init__(self):
        super().__init__()


class AtomPooling(nn.Module):
    def __init__(self, raw_feature_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        """
        Args:
            raw_feature_dim (int): Dimension of the raw residue-level features.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of linear layers for computing weights.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define a sequential module with multiple linear layers, GELU activation, and dropout
        layers = []
        input_dim = raw_feature_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, NUM_MAIN_SEQ_ATOMS))  # Final layer maps to NUM_MAIN_SEQ_ATOMS
        self.weight_layers = nn.Sequential(*layers)

    def forward(self, atom_encode: torch.Tensor, atom_mask: torch.Tensor, raw_feature: torch.Tensor) -> torch.Tensor:
        """
        Pool atom-level features into residue-level features.

        Args:
            atom_encode (torch.Tensor): Atom-level features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, hidden_dim).
            atom_mask (torch.Tensor): Atom mask of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            raw_feature (torch.Tensor): Residue-level raw features of shape (batch_size, max_len, raw_feature_dim).

        Returns:
            torch.Tensor: Residue-level pooled features of shape (batch_size, max_len, hidden_dim).
        """
        batch_size, max_len, raw_feature_dim = raw_feature.shape
        _, _, hidden_dim = atom_encode.shape

        # Reshape atom_mask and atom_encode to group by sequence units
        atom_mask = atom_mask.reshape(batch_size, max_len, NUM_MAIN_SEQ_ATOMS)
        atom_encode = atom_encode.reshape(batch_size, max_len, NUM_MAIN_SEQ_ATOMS, hidden_dim)

        # Compute weights from raw_feature using the sequential weight layers
        weights = self.weight_layers(raw_feature)  # Shape: (batch_size, max_len, NUM_MAIN_SEQ_ATOMS)

        weights = weights * atom_mask
        weights = F.softmax(weights, dim=-1)
        pooled_atom_features = torch.sum(atom_encode * weights.unsqueeze(-1), dim=2)
        return pooled_atom_features


class ResMPNN(nn.Module):
    def __init__(self):
        super().__init__()

class ResPooling(nn.Module):
    def __init__(self):
        super().__init__()

class Readout(nn.Module):
    def __init__(self):
        super().__init__()

class RNAMPNN(LightningModule):
    def __init__(self, num_neighbour: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.num_neighbour = num_neighbour

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]