import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from typing import Tuple, Any
from torch import Tensor

from ..config.glob import NUM_MAIN_SEQ_ATOMS, DEFAULT_HIDDEN_DIM, LEPS, SEPS, NUM_RES_TYPES
from torch.nn import functional as F

class GraphNormalization(nn.Module):
    def __init__(self, embedding_dim: int, epsilon: float = SEPS):
        """
        Graph normalization layer for normalizing node features in a graph.
        Args:
            embedding_dim (int): Dimension of the node features.
            epsilon (float): Small constant to avoid division by zero.
        """
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


def to_atom_format(coords: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    atom_mask = mask.unsqueeze(-1).expand(-1, -1, NUM_MAIN_SEQ_ATOMS).reshape(mask.shape[0], -1)
    return atom_coords, atom_mask


class AtomFeature(nn.Module):
    def __init__(self, num_atom_neighbours: int, atom_embedding_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.num_atom_neighbours = num_atom_neighbours
        self.atom_embedding_dim = atom_embedding_dim
        self.embedding = nn.Embedding(num_embeddings=NUM_MAIN_SEQ_ATOMS, embedding_dim=self.atom_embedding_dim)
        self.graph_norm = GraphNormalization(embedding_dim=self.atom_embedding_dim)

    def _get_atom_graph(self, atom_coords: torch.Tensor, atom_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a k-nearest neighbor graph for RNA atom coordinates.

        Args:
            atom_coords (torch.Tensor): Atom coordinates of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, 3).
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - atom_cross_dists: Distances to neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours).
                - atom_edge_index: Indices of neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours).
        """
        batch_size, num_atoms, _ = atom_coords.shape

        # Expand mask for pairwise distance calculation
        atom_mask_2d = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)  # (batch_size, num_atoms, num_atoms)

        # Compute pairwise distances
        d_coords = atom_coords.unsqueeze(1) - atom_coords.unsqueeze(2)  # (batch_size, num_atoms, num_atoms, 3)
        distances = torch.sqrt(torch.sum(d_coords ** 2, dim=-1) + SEPS)  # (batch_size, num_atoms, num_atoms)

        # Create a diagonal tensor with LEPS values
        diagonal = torch.eye(num_atoms, device=distances.device).unsqueeze(0)  # (1, num_atoms, num_atoms)
        distances = distances + diagonal * LEPS  # Add diagonal tensor to exclude self-loops

        # Mask invalid distances (real atoms to padding atoms)
        distances = distances * atom_mask_2d + (1.0 - atom_mask_2d) * LEPS  # Large value for invalid distances

        # Select k-nearest neighbors
        atom_cross_dists, atom_edge_idx = torch.topk(distances, min(self.num_atom_neighbours, num_atoms), dim=-1,
                                                     largest=False)

        # Handle cases where num_atom_neighbours exceeds the number of valid atoms
        if self.num_atom_neighbours > num_atoms:
            padding_size = self.num_atom_neighbours - num_atoms
            atom_cross_dists = torch.cat(
                [atom_cross_dists, torch.full((batch_size, num_atoms, padding_size), LEPS, device=distances.device)],
                dim=-1
            )
            atom_edge_idx = torch.cat(
                [atom_edge_idx,
                 torch.full((batch_size, num_atoms, padding_size), -1, device=distances.device, dtype=torch.long)],
                dim=-1
            )

        # Replace self-loops (where atom_edge_idx == node index) with -1
        node_indices = torch.arange(num_atoms, device=atom_edge_idx.device).view(1, -1, 1)  # Shape: (1, num_atoms, 1)
        atom_edge_idx = torch.where(atom_edge_idx == node_indices, -1, atom_edge_idx)

        # Set all neighbors of padding atoms to -1 and distances to LEPS
        padding_atom_mask = (atom_mask == 0).unsqueeze(-1)  # (batch_size, num_atoms, 1)
        atom_edge_idx = atom_edge_idx.masked_fill(padding_atom_mask, -1)
        atom_cross_dists = atom_cross_dists.masked_fill(padding_atom_mask, LEPS)

        return atom_cross_dists, atom_edge_idx


    def _atom_embedding(self, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate embedding vectors for atom types based on the cyclic pattern.

        Args:
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            torch.Tensor: Embedding vectors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
        """
        batch_size, num_atoms = atom_mask.shape

        # Generate cyclic atom types (0, 1, 2, 3, 4, 5, 6) for valid atoms
        atom_types = torch.arange(num_atoms, device=atom_mask.device) % NUM_MAIN_SEQ_ATOMS
        atom_types = atom_types.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, num_atoms)

        # Generate embeddings for atom types
        embeddings = self.embedding(atom_types)  # Shape: (batch_size, num_atoms, atom_embedding_dim)

        # Set embeddings for padding atoms (where atom_mask == 0) to zero
        embeddings = embeddings * atom_mask.unsqueeze(-1)

        # Normalize embeddings
        return self.graph_norm(embeddings, atom_mask)

    def forward(self, atom_coords: torch.Tensor, atom_mask: torch.Tensor):
        """
        Forward pass to process the coordinates and mask.
        Args:
            atom_coords (torch.Tensor): Atom coordinates of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, 3).
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            atom_embedding: Encoded atom features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
            atom_cross_dists: Distances to neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours).
            atom_edge_index: Indices of neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours).
        """
        atom_embedding = self._atom_embedding(atom_mask)
        atom_cross_dists, atom_edge_index = self._get_atom_graph(atom_coords, atom_mask)
        return atom_embedding, atom_cross_dists, atom_edge_index


class AtomMPNN(nn.Module):
    def __init__(self, atom_embedding_dim: int, depth_atom_mpnn: int, dropout: float = 0.1):
        """
        Message Passing Neural Network (MPNN) for atom-level features.
        Args:
            atom_embedding_dim (int): Dimension of the hidden layers.
            depth_atom_mpnn (int): Number of linear layers for message generation.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.graph_norm = GraphNormalization(embedding_dim=atom_embedding_dim)
        layers = []
        input_dim = atom_embedding_dim * 2 + 1
        for _ in range(depth_atom_mpnn):
            layers.append(nn.Linear(input_dim, atom_embedding_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = atom_embedding_dim
        self.message_layers = nn.Sequential(*layers)

    def message(self, atom_embedding: torch.Tensor, atom_cross_dists: torch.Tensor, atom_edge_index: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate messages for each edge in the graph.

        Args:
            atom_embedding (torch.Tensor): Node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
            atom_cross_dists (torch.Tensor): Distances to neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours).
            atom_edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours).
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
        Returns:
            messages (torch.Tensor): Messages of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours, atom_embedding_dim).
        """
        # Mask invalid nodes and expand encode for neighbors
        atom_embedding = atom_embedding * atom_mask.unsqueeze(-1)

        # Replace -1 indices with 0 (or any valid index) to avoid runtime errors
        safe_edge_index = atom_edge_index.clone()
        safe_edge_index[safe_edge_index == -1] = 0

        # Gather source features using the safe edge index
        source_features = torch.gather(
            atom_embedding.unsqueeze(2).expand(-1, -1, atom_edge_index.size(-1), -1),
            1,
            safe_edge_index.unsqueeze(-1).expand(-1, -1, -1, atom_embedding.shape[-1])
        )

        # Concatenate source, target, and distance features
        edge_features = torch.cat([
            source_features,
            atom_embedding.unsqueeze(2).expand(-1, -1, atom_edge_index.size(-1), -1),
            atom_cross_dists.unsqueeze(-1)
        ], dim=-1)

        # Generate messages and mask invalid ones
        valid_mask = (atom_edge_index != -1).unsqueeze(-1).float()
        messages = self.message_layers(edge_features) * valid_mask
        return messages

    @staticmethod
    def aggregation(atom_embedding: torch.Tensor, messages: torch.Tensor, atom_edge_index: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages for each node and update node features.

        Args:
            atom_embedding (torch.Tensor): Node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
            messages (torch.Tensor): Messages of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours, atom_embedding_dim).
            atom_edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours).
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            torch.Tensor: Updated node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
        """
        # Compute the sum of messages for each target node
        message_sum = torch.sum(messages, dim=2)

        # Compute the number of valid neighbors for each node
        valid_neighbors = (atom_edge_index != -1).sum(dim=-1, keepdim=True).float()
        valid_neighbors[valid_neighbors == 0] = 1  # Avoid division by zero

        # Compute the mean of messages
        aggregated_messages = message_sum / valid_neighbors

        # Add aggregated messages to the original node features
        updated_atom_embedding = atom_embedding + aggregated_messages

        # Mask invalid nodes (padding atoms)
        updated_atom_embedding = updated_atom_embedding * atom_mask.unsqueeze(-1)
        return updated_atom_embedding

    def forward(self, atom_embedding: torch.Tensor, atom_cross_dists: torch.Tensor, atom_edge_index: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AtomMPNN module.

        Args:
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            atom_embedding (torch.Tensor): Node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
            atom_cross_dists (torch.Tensor): Distances to neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours).
            atom_edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbours).

        Returns:
            atom_embedding (torch.Tensor): Updated node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
        """
        messages = self.message(atom_embedding, atom_cross_dists, atom_edge_index, atom_mask)
        atom_embedding = self.aggregation(atom_embedding, messages, atom_edge_index, atom_mask)
        atom_embedding = self.graph_norm(atom_embedding, atom_mask)
        return atom_embedding


class AtomPooling(nn.Module):
    def __init__(self, raw_dim: int, atom_pooling_hidden_dim: int, num_layers: int, dropout: float):
        """
        Args:
            raw_dim (int): Dimension of the raw residue-level features.
            atom_pooling_hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of linear layers for computing weights.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.num_layers = num_layers
        self.atom_pooling_hidden_dim = atom_pooling_hidden_dim

        layers = []
        input_dim = raw_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, atom_pooling_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = atom_pooling_hidden_dim
        layers.append(nn.Linear(atom_pooling_hidden_dim, NUM_MAIN_SEQ_ATOMS))
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


class ResFeature(nn.Module):
    def __init__(self,
                 num_neighbours: int,
                 num_inside_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_inside_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_inside_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS-1,
                 num_cross_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_cross_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 atom_pooling_hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 atom_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 res_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 res_edge_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 num_atom_pooling_layers: int = 2,
                 num_layers: int = 2,
                 num_edge_layers: int = 2,
                 dropout: float = 0.1):
        """
        Residue-level feature extraction module.
        Args:
            num_neighbours (int): Number of neighbors for each residue.
            num_inside_dist_atoms (int): Number of atoms for distance calculation.
            num_inside_angle_atoms (int): Number of atoms for angle calculation.
            num_inside_dihedral_atoms (int): Number of atoms for dihedral angle calculation.
            num_cross_dist_atoms (int): Number of atoms for cross distance calculation.
            num_cross_angle_atoms (int): Number of atoms for cross angle calculation.
            num_cross_dihedral_atoms (int): Number of atoms for cross dihedral angle calculation.
            atom_pooling_hidden_dim (int): Dimension of the hidden layers for atom pooling.
            atom_embedding_dim (int): Dimension of the atom-level features.
            res_embedding_dim (int): Dimension of the residue-level node features.
            res_edge_embedding_dim (int): Dimension of the residue-level edge features.
            num_atom_pooling_layers (int): Number of linear layers for atom pooling.
            num_layers (int): Number of linear layers for residue-level node features.
            num_edge_layers (int): Number of linear layers for residue-level edge features.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.num_neighbours = num_neighbours
        assert num_inside_dist_atoms >= 2, f"num_inside_dist_atoms({num_inside_dist_atoms}) must be at least 2 for distance calculation."
        self.num_inside_dist_atoms = num_inside_dist_atoms
        assert num_inside_angle_atoms >= 3, f"num_inside_angle_atoms({num_inside_angle_atoms}) must be at least 3 for angle calculation."
        self.num_inside_angle_atoms = num_inside_angle_atoms
        assert num_inside_dihedral_atoms >= 4, f"num_inside_dihedral_atoms({num_inside_dihedral_atoms}) must be at least 4 for dihedral calculation."
        self.num_inside_dihedral_atoms = num_inside_dihedral_atoms
        assert num_cross_dist_atoms >= 2, f"num_cross_dist_atoms({num_cross_dist_atoms}) must be at least 2 for cross distance calculation."
        self.num_cross_dist_atoms = num_cross_dist_atoms
        assert num_cross_angle_atoms >= 3, f"num_cross_angle_atoms({num_cross_angle_atoms}) must be at least 3 for cross angle calculation."
        self.num_cross_angle_atoms = num_cross_angle_atoms
        assert num_cross_dihedral_atoms >= 4, f"num_cross_dihedral_atoms({num_cross_dihedral_atoms}) must be at least 4 for cross dihedral calculation."
        self.num_cross_dihedral_atoms = num_cross_dihedral_atoms
        raw_dim = num_inside_dist_atoms - 1 + num_inside_angle_atoms - 2 + num_inside_dihedral_atoms - 3
        raw_edge_dim = num_cross_dist_atoms ** 2 + (num_cross_angle_atoms - 1) ** 2 + (num_cross_dihedral_atoms - 2) ** 2
        self.atom_pooling = AtomPooling(raw_dim=raw_dim,
                                        atom_pooling_hidden_dim=atom_pooling_hidden_dim,
                                        num_layers=num_atom_pooling_layers,
                                        dropout=dropout)

        layers = []
        input_dim = raw_dim + atom_embedding_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, res_embedding_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = res_embedding_dim
        self.res_embedding_layers = nn.Sequential(*layers)
        self.graph_norm = GraphNormalization(embedding_dim=res_embedding_dim)

        layers = []
        input_dim = raw_edge_dim
        for _ in range(num_edge_layers):
            layers.append(nn.Linear(input_dim, res_edge_embedding_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = res_edge_embedding_dim
        self.res_edge_embedding_layers = nn.Sequential(*layers)

    def _get_res_graph(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate residue-level graph structure without including the node itself as a neighbor.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).

        Returns:
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len, num_neighbours).
        """
        batch_size, max_len, num_atoms, _ = coords.shape

        # Compute average coordinates for each residue
        avg_coords = coords.mean(dim=2)  # Shape: (batch_size, max_len, 3)

        # Compute pairwise distances between residues
        d_coords = avg_coords.unsqueeze(1) - avg_coords.unsqueeze(2)  # Shape: (batch_size, max_len, max_len, 3)
        distances = torch.sqrt(torch.sum(d_coords ** 2, dim=-1) + SEPS)  # Shape: (batch_size, max_len, max_len)

        # Mask invalid distances (real residues to padding residues)
        residue_mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # Shape: (batch_size, max_len, max_len)
        distances = distances * residue_mask_2d + (1.0 - residue_mask_2d) * LEPS

        # Exclude self-loops by setting diagonal elements to a large value
        diagonal_mask = torch.eye(max_len, device=distances.device).unsqueeze(0)  # Shape: (1, max_len, max_len)
        distances = distances + diagonal_mask * LEPS

        # Select k-nearest neighbors
        _, edge_index = torch.topk(
            distances,
            min(self.num_neighbours, max_len),  # Ensure we don't request more neighbors than available
            dim=-1,
            largest=False
        )

        # If padding is needed (when max_len < self.num_neighbours)
        if self.num_neighbours > max_len:
            padding_size = self.num_neighbours - max_len
            # Pad `edge_index` with -1 and `dist_neighbour` with a large value (e.g., LEPS)
            edge_index = torch.cat(
                [edge_index,
                torch.full((batch_size, max_len, padding_size), -1, device=edge_index.device, dtype=edge_index.dtype)],
                dim=-1
            )

        # Create a tensor of indices for comparison
        indices = torch.arange(max_len, device=edge_index.device).view(1, -1, 1)  # Shape: (1, max_len, 1)

        # Replace self-loops (where edge_index == indices) with -1
        edge_index = torch.where(edge_index == indices, -1, edge_index)

        # Handle cases where num_neighbours exceeds the number of valid residues
        valid_neighbors = (residue_mask_2d.sum(dim=-1) - 1).clamp(min=0)  # Exclude self
        padding_mask = valid_neighbors.unsqueeze(-1) < torch.arange(self.num_neighbours, device=distances.device)

        edge_index = edge_index.masked_fill(padding_mask, -1)  # Fill remaining neighbors with -1

        # Set all neighbors of padding residues to -1
        padding_residue_mask = (mask == 0).unsqueeze(-1)  # (batch_size, max_len, 1)
        edge_index = edge_index.masked_fill(padding_residue_mask, -1)

        return edge_index

    @staticmethod
    def _gather_neighbours(coords: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Gather neighbor coordinates based on edge indices and form a tensor of shape
        (batch_size, max_len, num_neighbours, NUM_MAIN_SEQ_ATOMS, 3).

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            edge_index (torch.Tensor): Edge indices of shape (batch_size, max_len, num_neighbours).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).

        Returns:
            neighbours_coords (torch.Tensor): Tensor of shape (batch_size, max_len, num_neighbours, NUM_MAIN_SEQ_ATOMS, 3).
        """
        _, _, num_atoms, _ = coords.shape
        num_neighbours = edge_index.shape[-1]

        # Replace -1 in edge_index with 0 temporarily
        safe_edge_index = edge_index.clone()
        safe_edge_index[safe_edge_index == -1] = 0

        # Gather neighbor coordinates
        neighbours_coords = torch.gather(
            coords.unsqueeze(2).expand(-1, -1, num_neighbours, -1, -1),
            # Shape: (batch_size, max_len, num_neighbours, NUM_MAIN_SEQ_ATOMS, 3)
            dim=1,
            index=safe_edge_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, num_atoms, 3)
            # Shape: (batch_size, max_len, num_neighbours, NUM_MAIN_SEQ_ATOMS, 3)
        )

        # Mask invalid neighbors (where edge_index == -1)
        invalid_mask = (edge_index == -1).unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, max_len, num_neighbours, 1, 1)
        neighbours_coords = neighbours_coords.masked_fill(invalid_mask, LEPS)

        # Mask padding nodes and their neighbors
        padding_mask = (mask == 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, max_len, 1, 1, 1)
        neighbours_coords = neighbours_coords.masked_fill(padding_mask, LEPS)

        return neighbours_coords

    def _inside_dists(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between all atoms within each sequence unit.

        Args:
            coords (torch.Tensor): Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask of shape (batch_size, max_len).

        Returns:
            inside_dists (torch.Tensor): Pairwise distances of shape (batch_size, max_len, num_inside_dist_atoms - 1).
        """
        _, _, num_atoms, _ = coords.shape
        assert num_atoms >= self.num_inside_dist_atoms, f"NUM_MAIN_SEQ_ATOMS({num_atoms}) must be at least num_inside_angle_atoms({self.num_inside_angle_atoms}) for angle calculation."

        # Truncate to the first `num_inside_dist_atoms`
        coords = coords[:, :, :self.num_inside_dist_atoms, :]  # Shape: (batch_size, max_len, num_inside_dist_atoms, 3)

        # Compute vectors between adjacent atoms
        vecs = coords[:, :, 1:] - coords[:, :, :-1]  # Shape: (batch_size, max_len, num_inside_dist_atoms-1, 3)

        # Compute distances
        inside_dists = torch.sqrt(
            torch.sum(vecs ** 2, dim=-1) + SEPS)  # Shape: (batch_size, max_len, num_inside_dist_atoms-1)

        # Replace distances for padding residues with 1e6
        padding_mask = (mask == 0).unsqueeze(-1)  # Shape: (batch_size, max_len, 1)
        inside_dists = inside_dists.masked_fill(padding_mask, 1e6)

        return inside_dists

    def _inside_angles(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine of angles formed by every three consecutive atoms in each residue.

        Args:
            coords (torch.Tensor): Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask of shape (batch_size, max_len).

        Returns:
            inside_angles (torch.Tensor): Cosine values of angles of shape (batch_size, max_len, num_inside_angle_atoms-2).
        """
        _, _, num_atoms, _ = coords.shape
        assert num_atoms >= self.num_inside_angle_atoms, f"NUM_MAIN_SEQ_ATOMS({num_atoms}) must be at least num_inside_angle_atoms({self.num_inside_angle_atoms}) for angle calculation."

        # Truncate to the first six atoms
        coords = coords[:, :, :self.num_inside_angle_atoms, :]  # Shape: (batch_size, max_len, num_inside_angle_atoms, 3)

        # Compute vectors between adjacent atoms
        vecs = coords[:, :, 1:] - coords[:, :, :-1]  # Shape: (batch_size, max_len, num_inside_angle_atoms-1, 3)

        # Compute dot products between consecutive vectors
        dot_products = (vecs[:, :, :-1] * vecs[:, :, 1:]).sum(dim=-1)  # Shape: (batch_size, max_len, num_inside_angle_atoms-2)

        # Compute norms of the vectors
        norms = torch.norm(vecs, dim=-1)  # Shape: (batch_size, max_len, num_inside_angle_atoms-1)

        # Compute cosine values for the angles
        inside_angles = dot_products / (norms[:, :, :-1] * norms[:, :, 1:] + SEPS)  # Shape: (batch_size, max_len, num_inside_angle_atoms-2)

        # Mask invalid residues (set cosine values to zero for padding residues)
        inside_angles = inside_angles * mask.unsqueeze(-1)

        return inside_angles

    def _inside_dihedrals(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute dihedral angles formed by every four consecutive atoms in each residue.

        Args:
            coords (torch.Tensor): Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask of shape (batch_size, max_len).

        Returns:
            inside_dihedrals (torch.Tensor): Cosine of dihedral angles of shape (batch_size, max_len, num_inside_dihedral_atoms-3).
        """
        _, _, num_atoms, _ = coords.shape
        assert num_atoms >= self.num_inside_dihedral_atoms, f"NUM_MAIN_SEQ_ATOMS({num_atoms}) must be at least num_inside_dihedral_atoms({self.num_inside_dihedral_atoms}) for dihedral calculation."

        # Truncate to the first num_inside_dihedral_atoms
        coords = coords[:, :, :self.num_inside_dihedral_atoms, :]  # Shape: (batch_size, max_len, num_inside_dihedral_atoms, 3)

        # Compute vectors between consecutive atoms
        vec1 = F.normalize(coords[:, :, 1:, :] - coords[:, :, :-1, :], dim=-1, eps=SEPS)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-1, 3)
        vec2 = vec1[:, :, 1:, :]  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-2, 3)
        vec1 = vec1[:, :, :-1, :]  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-2, 3)

        normal = F.normalize(torch.linalg.cross(vec1, vec2), dim=-1, eps=SEPS) # Shape: (batch_size, max_len, num_inside_dihedral_atoms-2, 3)
        inside_dihedrals = (normal[:, :, 1:, :] * normal[:, :, :-1, :]).sum(dim=-1)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-3)
        inside_dihedrals = inside_dihedrals * mask.unsqueeze(-1)
        return inside_dihedrals

    def _cross_dists(self, coords: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between `num_cross_dist_atoms` atoms of the central residue
        and its neighboring residues, with special handling for padding nodes.

        Args:
            coords (torch.Tensor): Tensor of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3) containing atom coordinates.
            mask (torch.Tensor): Tensor of shape (batch_size, max_len) indicating valid residues.
            edge_index (torch.Tensor): Tensor of shape (batch_size, max_len, num_neighbour) containing neighbor indices.

        Returns:
            cross_dists (torch.Tensor): Tensor of shape (batch_size, max_len, num_neighbour, num_cross_dist_atoms ** 2) containing pairwise distances.
        """
        batch_size, max_len, _, _, = coords.shape
        num_neighbour = edge_index.shape[-1]
        num_cross_dist_atoms = self.num_cross_dist_atoms

        # Truncate to the first `num_cross_dist_atoms`
        coords = coords[:, :, :num_cross_dist_atoms, :]  # Shape: (batch_size, max_len, num_cross_dist_atoms, 3)

        # Replace invalid indices (-1) in edge_index with 0 for gathering
        edge_index = edge_index.clone()
        invalid_mask = edge_index == -1
        edge_index[invalid_mask] = 0

        # Gather neighbor coordinates
        neighbour_coords = self._gather_neighbours(coords, edge_index, mask)  # Shape: (batch_size, max_len, num_neighbour, num_cross_dist_atoms, 3)
        # Compute pairwise distances
        central_coords = coords.unsqueeze(2)  # Shape: (batch_size, max_len, 1, num_cross_dist_atoms, 3)
        diffs = central_coords.unsqueeze(4) - neighbour_coords.unsqueeze(3)  # Shape: (batch_size, max_len, num_neighbour, num_cross_dist_atoms, num_cross_dist_atoms, 3)
        dists = torch.sqrt((diffs ** 2).sum(dim=-1) + SEPS)  # Shape: (batch_size, max_len, num_neighbour, num_cross_dist_atoms, num_cross_dist_atoms)

        cross_dists = dists.view(batch_size, max_len, num_neighbour, -1)

        valid_mask = mask.unsqueeze(-1).unsqueeze(-1) * (~invalid_mask).unsqueeze(-1)  # Shape: (batch_size, max_len, num_neighbour, 1)
        cross_dists = cross_dists * valid_mask + (1 - valid_mask) * LEPS

        return cross_dists

    def _cross_angles(self, coords: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine of angles formed by every three consecutive atoms in each residue
        Args:
            coords (torch.Tensor): Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask of shape (batch_size, max_len).
            edge_index (torch.Tensor): Edge indices of shape (batch_size, max_len, num_neighbours).
        Returns:
            cross_angles (torch.Tensor): Cosine values of angles of shape (batch_size, max_len, num_neighbours, (num_cross_angle_atoms-1) ** 2).
        """
        batch_size, max_len, _, _ = coords.shape
        num_neighbour = edge_index.shape[-1]
        num_cross_angles_atoms = self.num_cross_angle_atoms

        # Truncate to the first `num_cross_angles_atoms`
        coords = coords[:, :, :num_cross_angles_atoms, :]  # Shape: (batch_size, max_len, num_cross_angles_atoms, 3)

        # Replace invalid indices (-1) in edge_index with 0 for gathering
        edge_index = edge_index.clone()
        invalid_mask = edge_index == -1
        edge_index[invalid_mask] = 0

        # Gather neighbor coordinates
        neighbour_coords = self._gather_neighbours(coords, edge_index,mask)  # Shape: (batch_size, max_len, num_neighbour, num_cross_angles_atoms, 3)

        # Compute vectors for sequential edges
        central_vectors = coords[:, :, 1:, :] - coords[:, :, :-1, :]  # Shape: (batch_size, max_len, num_cross_angles_atoms-1, 3)
        neighbour_vectors = neighbour_coords[:, :, :, 1:, :] - neighbour_coords[:, :, :, :-1, :]  # Shape: (batch_size, max_len, num_neighbour, num_cross_angles_atoms-1, 3)

        # Normalize vectors
        central_norm = F.normalize(central_vectors, dim=-1)  # Shape: (batch_size, max_len, num_cross_angles_atoms-1, 3)
        neighbour_norm = F.normalize(neighbour_vectors, dim=-1)  # Shape: (batch_size, max_len, num_neighbour, num_cross_angles_atoms-1, 3)


        central_norm = central_norm.unsqueeze(2).unsqueeze(4)  # Shape: (batch_size, max_len, 1, num_cross_angles_atoms-1, 1, 3)
        neighbour_norm = neighbour_norm.unsqueeze(3)  # Shape: (batch_size, max_len, num_neighbour, 1, num_cross_angles_atoms-1, 3)

        dot_products = (central_norm * neighbour_norm).sum(dim=-1)  # Shape: (batch_size, max_len, num_neighbour, num_cross_angles_atoms-1, num_cross_angles_atoms-1)

        cross_angles = dot_products.view(batch_size, max_len, num_neighbour, -1)

        valid_mask = mask.unsqueeze(-1).unsqueeze(-1) * (~invalid_mask).unsqueeze(-1)  # Shape: (batch_size, max_len, num_neighbour, 1)
        cross_angles = cross_angles * valid_mask

        return cross_angles

    def _cross_dihedrals(self, coords: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine of dihedral angles formed by every four consecutive atoms in each residue
        Args:
            coords (torch.Tensor): Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask of shape (batch_size, max_len).
            edge_index (torch.Tensor): Edge indices of shape (batch_size, max_len, num_neighbours).
        Returns:
            cross_dihedrals (torch.Tensor): Cosine values of dihedral angles of shape (batch_size, max_len, num_neighbours, (num_cross_dihedral_atoms-2) ** 2).
        """
        batch_size, max_len, _, _ = coords.shape
        num_neighbour = edge_index.shape[-1]
        num_cross_dihedrals = self.num_cross_dihedral_atoms

        coords = coords[:, :, :num_cross_dihedrals, :]  # Shape: (batch_size, max_len, num_cross_dihedrals, 3)

        edge_index = edge_index.clone()
        invalid_mask = edge_index == -1
        edge_index[invalid_mask] = 0

        neighbour_coords = self._gather_neighbours(coords, edge_index, mask)  # Shape: (batch_size, max_len, num_neighbour, num_cross_dihedrals, 3)

        central_vectors = coords[:, :, 1:, :] - coords[:, :, :-1, :]  # Shape: (batch_size, max_len, num_cross_dihedrals-1, 3)
        neighbour_vectors = neighbour_coords[:, :, :, 1:, :] - neighbour_coords[:, :, :, :-1, :]  # Shape: (batch_size, max_len, num_neighbour, num_cross_dihedrals-1, 3)

        central_normals = F.normalize(
            torch.cross(central_vectors[:, :, :-1, :], central_vectors[:, :, 1:, :], dim=-1),
            dim=-1,
            eps=SEPS
        )  # Shape: (batch_size, max_len, num_cross_dihedrals-2, 3)
        neighbour_normals = F.normalize(
            torch.cross(neighbour_vectors[:, :, :, :-1, :], neighbour_vectors[:, :, :, 1:, :], dim=-1),
            dim=-1,
            eps=SEPS
        )  # Shape: (batch_size, max_len, num_neighbour, num_cross_dihedrals-2, 3)

        central_normals = central_normals.unsqueeze(2).unsqueeze(4)  # Shape: (batch_size, max_len, 1, num_cross_dihedrals-2, 1, 3)
        neighbour_normals = neighbour_normals.unsqueeze(3)  # Shape: (batch_size, max_len, num_neighbour, 1, num_cross_dihedrals-2, 3)

        dot_products = (central_normals * neighbour_normals).sum(dim=-1)  # Shape: (batch_size, max_len, num_neighbour, num_cross_dihedrals-2, num_cross_dihedrals-2)

        cross_dihedrals = dot_products.view(batch_size, max_len, num_neighbour, -1)

        valid_mask = mask.unsqueeze(-1).unsqueeze(-1) * (~invalid_mask).unsqueeze(-1)  # Shape: (batch_size, max_len, num_neighbour, 1)
        cross_dihedrals = cross_dihedrals * valid_mask

        return cross_dihedrals

    def _res_embedding(self, coords: torch.Tensor, mask: torch.Tensor, atom_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute residue-level graph node features.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).
            atom_embedding (torch.Tensor): Atom-level embeddings of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).

        Returns:
            res_embedding (torch.Tensor): Residue embedding of shape (batch_size, max_len, res_embedding_dim).
        """
        inside_dists = self._inside_dists(coords, mask)  # Shape: (batch_size, max_len, num_inside_dist_atoms * (num_inside_dist_atoms - 1) / 2)
        inside_angles = self._inside_angles(coords, mask)  # Shape: (batch_size, max_len, num_inside_angle_atoms - 2)
        inside_dihedrals = self._inside_dihedrals(coords, mask)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms - 3)
        atom_mask = mask.unsqueeze(-1).expand(-1, -1, NUM_MAIN_SEQ_ATOMS).reshape(mask.shape[0], -1)

        raw = torch.cat([inside_dists, inside_angles, inside_dihedrals], dim=-1)  # Shape: (batch_size, max_len, raw_dim)
        pooled_atom_embedding = self.atom_pooling(atom_embedding, atom_mask, raw)  # Shape: (batch_size, max_len, atom_embedding_dim)
        res_embedding = self.res_embedding_layers(torch.cat([raw, pooled_atom_embedding],dim=-1))  # Shape: (batch_size, max_len, res_embedding_dim)
        res_embedding = res_embedding * mask.unsqueeze(-1)

        return res_embedding

    def _res_edge_embedding(self, coords: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute residue-level edge features.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len, num_neighbours).

        Returns:
            torch.Tensor: Residue edge embedding of shape (batch_size, max_len, num_neighbours, res_edge_embedding_dim).
        """
        # Compute edge features
        cross_dists = self._cross_dists(coords, mask, edge_index)  # Shape: (batch_size, max_len, num_neighbours, num_cross_dist_atoms * (num_cross_dist_atoms - 1) / 2)
        cross_angles = self._cross_angles(coords, mask, edge_index)  # Shape: (batch_size, max_len, num_neighbours, (num_cross_angle_atoms - 1) * (num_cross_angle_atoms - 2) / 2)
        cross_dihedrals = self._cross_dihedrals(coords, mask, edge_index)  # Shape: (batch_size, max_len, num_neighbours, (num_cross_dihedral_atoms - 2) * (num_cross_dihedral_atoms - 3) / 2)

        # Concatenate raw edge features
        raw_edge_features = torch.cat([cross_dists, cross_angles, cross_dihedrals], dim=-1)  # Shape: (batch_size, max_len, num_neighbours, raw_edge_dim)

        # Compute edge embeddings
        res_edge_embedding = self.res_edge_embedding_layers(raw_edge_features)  # Shape: (batch_size, max_len, num_neighbours, res_edge_embedding_dim)

        # Create masks for invalid edges and padding nodes
        invalid_edge_mask = (edge_index == -1).unsqueeze(-1)  # Shape: (batch_size, max_len, num_neighbours, 1)
        padding_node_mask = (mask == 0).unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, max_len, 1, 1)

        # Set embeddings of invalid edges and padding-related edges to zero
        res_edge_embedding = res_edge_embedding.masked_fill(invalid_edge_mask, 0.0)
        res_edge_embedding = res_edge_embedding.masked_fill(padding_node_mask, 0.0)

        return res_edge_embedding

    def forward(self, coords: torch.Tensor, mask: torch.Tensor, atom_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method to compute residue-level graph embeddings.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).
            atom_embedding (torch.Tensor): Atom-level embeddings of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Residue node embeddings, edge embeddings, and edge indices.
        """
        # Generate residue graph edge information
        edge_index = self._get_res_graph(coords, mask)  # Shape: (batch_size, max_len, num_neighbours)

        # Compute residue edge embeddings
        res_edge_embedding = self._res_edge_embedding(coords, mask, edge_index)  # Shape: (batch_size, max_len, num_neighbours, res_edge_embedding_dim)

        # Compute residue node embeddings
        res_embedding = self._res_embedding(coords, mask, atom_embedding)  # Shape: (batch_size, max_len, res_embedding_dim)
        res_embedding = self.graph_norm(res_embedding, mask)

        return res_embedding, res_edge_embedding, edge_index


class ResMPNN(nn.Module):
    def __init__(self,
                 res_embedding_dim: int,
                 res_edge_embedding_dim: int,
                 depth_res_mpnn: int,
                 num_edge_layers: int,
                 dropout: float = 0.1):
        """
        Initialize the Residue Message Passing Neural Network (ResMPNN).
        Args:
            res_embedding_dim: The dimension of the residue embedding.
            res_edge_embedding_dim: The dimension of the residue edge embedding.
            depth_res_mpnn: The number of message passing layers.
            num_edge_layers: The number of edge update layers.
            dropout: The dropout rate for regularization.
        """
        super().__init__()
        self.graph_norm = GraphNormalization(embedding_dim=res_embedding_dim)

        # Message passing layers for node updates
        layers = []
        input_dim = res_embedding_dim * 2 + res_edge_embedding_dim
        for _ in range(depth_res_mpnn):
            layers.append(nn.Linear(input_dim, res_embedding_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = res_embedding_dim
        self.message_layers = nn.Sequential(*layers)

        # Edge update layers
        layers = []
        input_dim = res_embedding_dim * 2 + res_edge_embedding_dim
        for _ in range(num_edge_layers):
            layers.append(nn.Linear(input_dim, res_edge_embedding_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = res_edge_embedding_dim
        self.edge_layers = nn.Sequential(*layers)

    def message(self, res_embedding: torch.Tensor, res_edge_embedding: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Generate messages for each edge in the residue graph.

        Args:
            res_embedding (torch.Tensor): Node features of shape (batch_size, max_len, res_embedding_dim).
            res_edge_embedding (torch.Tensor): Edge features of shape (batch_size, max_len, num_neighbours, res_edge_embedding_dim).
            edge_index (torch.Tensor): Edge indices of shape (batch_size, max_len, num_neighbours).
            mask (torch.Tensor): Mask indicating valid residues of shape (batch_size, max_len).

        Returns:
            torch.Tensor: Messages of shape (batch_size, max_len, num_neighbours, res_embedding_dim).
        """
        _, _, num_neighbours, _ = res_edge_embedding.shape

        # Mask invalid nodes and expand for neighbors
        res_embedding = res_embedding * mask.unsqueeze(-1)

        # Replace invalid indices (-1) with 0 to avoid runtime errors
        safe_edge_index = edge_index.clone()
        safe_edge_index[safe_edge_index == -1] = 0

        # Gather neighbor features using the safe edge index
        neighbor_features = torch.gather(
            res_embedding.unsqueeze(2).expand(-1, -1, num_neighbours, -1),
            dim=1,
            index=safe_edge_index.unsqueeze(-1).expand(-1, -1, -1, res_embedding.shape[-1])
        )  # Shape: (batch_size, max_len, num_neighbours, res_embedding_dim)

        # Concatenate source, target, and edge features
        edge_inputs = torch.cat([
            res_embedding.unsqueeze(2).expand(-1, -1, num_neighbours, -1),  # Source node features
            neighbor_features,  # Target node features
            res_edge_embedding  # Edge features
        ], dim=-1)  # Shape: (batch_size, max_len, num_neighbours, res_embedding_dim * 2 + res_edge_embedding_dim)

        # Generate messages and mask invalid ones
        valid_mask = (edge_index != -1).unsqueeze(-1).float()  # Shape: (batch_size, max_len, num_neighbours, 1)
        messages = self.message_layers(edge_inputs) * valid_mask  # Shape: (batch_size, max_len, num_neighbours, res_embedding_dim)

        return messages

    @staticmethod
    def aggregation(mask: torch.Tensor, res_embedding: torch.Tensor, messages: torch.Tensor,
                    edge_index: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages for each node and update node features.

        Args:
            mask (torch.Tensor): Mask indicating valid residues of shape (batch_size, max_len).
            res_embedding (torch.Tensor): Node features of shape (batch_size, max_len, res_embedding_dim).
            messages (torch.Tensor): Messages of shape (batch_size, max_len, num_neighbours, res_embedding_dim).
            edge_index (torch.Tensor): Edge indices of shape (batch_size, max_len, num_neighbours).

        Returns:
            torch.Tensor: Updated node features of shape (batch_size, max_len, res_embedding_dim).
        """
        # Compute the sum of messages for each target node
        message_sum = torch.sum(messages, dim=2)  # Shape: (batch_size, max_len, res_embedding_dim)

        # Compute the number of valid neighbors for each node
        valid_neighbors = (edge_index != -1).sum(dim=-1, keepdim=True).float()  # Shape: (batch_size, max_len, 1)
        valid_neighbors[valid_neighbors == 0] = 1  # Avoid division by zero

        # Compute the mean of messages
        aggregated_messages = message_sum / valid_neighbors  # Shape: (batch_size, max_len, res_embedding_dim)

        # Add aggregated messages to the original node features
        updated_res_embedding = res_embedding + aggregated_messages  # Shape: (batch_size, max_len, res_embedding_dim)

        # Mask invalid nodes (padding residues)
        updated_res_embedding = updated_res_embedding * mask.unsqueeze(-1)  # Shape: (batch_size, max_len, res_embedding_dim)

        return updated_res_embedding

    def _update_edges(self, res_embedding: torch.Tensor, res_edge_embedding: torch.Tensor,
                      edge_index: torch.Tensor) -> torch.Tensor:
        """
        Update edge features in the residue graph.

        Args:
            res_embedding (torch.Tensor): Node features of shape (batch_size, max_len, res_embedding_dim).
            res_edge_embedding (torch.Tensor): Edge features of shape (batch_size, max_len, num_neighbours, res_edge_embedding_dim).
            edge_index (torch.Tensor): Edge indices of shape (batch_size, max_len, num_neighbours).

        Returns:
            torch.Tensor: Updated edge features of shape (batch_size, max_len, num_neighbours, res_edge_embedding_dim).
        """
        # Gather neighbor node features using edge_index
        _, _, num_neighbours = edge_index.shape
        res_embedding_dim = res_embedding.shape[-1]

        # Replace -1 in edge_index with 0 temporarily for safe indexing
        safe_edge_index = edge_index.clone()
        safe_edge_index[safe_edge_index == -1] = 0

        # Gather neighbor node features
        neighbor_features = torch.gather(
            res_embedding.unsqueeze(2).expand(-1, -1, num_neighbours, -1),
            # Shape: (batch_size, max_len, num_neighbours, res_embedding_dim)
            dim=1,
            index=safe_edge_index.unsqueeze(-1).expand(-1, -1, -1, res_embedding_dim)
        )

        # Concatenate central node features, neighbor node features, and edge features
        central_features = res_embedding.unsqueeze(2).expand(-1, -1, num_neighbours, -1)  # Shape: (batch_size, max_len, num_neighbours, res_embedding_dim)
        concatenated_features = torch.cat([central_features, neighbor_features, res_edge_embedding], dim=-1)  # Shape: (batch_size, max_len, num_neighbours, res_embedding_dim * 2 + res_edge_embedding_dim)

        # Update edge features using self.edge_layers
        updated_edge_features = self.edge_layers(concatenated_features)  # Shape: (batch_size, max_len, num_neighbours, res_edge_embedding_dim)

        return updated_edge_features

    def forward(self, res_embedding: torch.Tensor, res_edge_embedding: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for residue-level MPNN.

        Args:
            res_embedding (torch.Tensor): Node features of shape (batch_size, max_len, res_embedding_dim).
            res_edge_embedding (torch.Tensor): Edge features of shape (batch_size, max_len, num_neighbours, res_edge_embedding_dim).
            edge_index (torch.Tensor): Edge indices of shape (batch_size, max_len, num_neighbours).
            mask (torch.Tensor): Mask indicating valid residues of shape (batch_size, max_len).

        Returns:
            Tuple[Tensor, Tensor]:
                - res_embedding: Updated node features of shape (batch_size, max_len, res_embedding_dim).
                - res_edge_embedding: Updated edge features of shape (batch_size, max_len, num_neighbours, res_edge_embedding_dim).
        """
        # Generate messages
        messages = self.message(res_embedding, res_edge_embedding, edge_index, mask)

        # Aggregate messages to update node features
        res_embedding = self.aggregation(mask, res_embedding, messages, edge_index)

        # Apply graph normalization
        res_embedding = self.graph_norm(res_embedding, mask)

        # Update edge features
        res_edge_embedding = self._update_edges(res_embedding, res_edge_embedding, edge_index)

        return res_embedding, res_edge_embedding


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
            torch.Tensor: Predicted residue type logits of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS).
        """
        # Apply feedforward network
        logits = self.readout_layers(res_embedding)  # Shape: (batch_size, max_len, NUM_MAIN_SEQ_ATOMS)

        # Mask padding residues
        logits = logits * mask.unsqueeze(-1)  # Set predictions for padding residues to zero

        return logits


class RNAMPNN(LightningModule):
    def __init__(self,
                 num_atom_neighbours: int = 30,
                 atom_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 depth_atom_mpnn: int = 2,
                 num_atom_mpnn_layers = 2,
                 num_res_neighbours: int = 30,
                 num_inside_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_inside_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_inside_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_cross_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 atom_pooling_hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 res_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 res_edge_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 num_atom_pooling_layers: int = 2,
                 depth_res_feature: int = 2,
                 depth_res_edge_feature: int = 2,
                 num_res_mpnn_layers: int = 2,
                 depth_res_mpnn: int = 2,
                 num_mpnn_edge_layers: int = 2,
                 readout_hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 num_readout_layers: int = 2,
                 dropout: float = 0.1,
                 lr: float = 1e-3,):
        super().__init__()
        self.save_hyperparameters()
        self.atom_feature = AtomFeature(num_atom_neighbours, atom_embedding_dim)
        self.atom_mpnn_layers = nn.ModuleList([AtomMPNN(atom_embedding_dim, depth_atom_mpnn, dropout=dropout) for _ in range(num_atom_mpnn_layers)])
        self.res_feature = ResFeature(num_neighbours=num_res_neighbours,
                                      num_inside_dist_atoms=num_inside_dist_atoms,
                                      num_inside_angle_atoms=num_inside_angle_atoms,
                                      num_inside_dihedral_atoms=num_inside_dihedral_atoms,
                                      num_cross_dist_atoms=num_cross_dist_atoms,
                                      num_cross_angle_atoms=num_cross_angle_atoms,
                                      num_cross_dihedral_atoms=num_cross_dihedral_atoms,
                                      atom_pooling_hidden_dim=atom_pooling_hidden_dim,
                                      atom_embedding_dim=atom_embedding_dim,
                                      res_embedding_dim=res_embedding_dim,
                                      res_edge_embedding_dim=res_edge_embedding_dim,
                                      num_atom_pooling_layers=num_atom_pooling_layers,
                                      num_layers=depth_res_feature,
                                      num_edge_layers=depth_res_edge_feature,
                                      dropout=dropout)
        self.res_mpnn_layers = nn.ModuleList([ResMPNN(res_embedding_dim=res_embedding_dim, res_edge_embedding_dim=res_edge_embedding_dim, depth_res_mpnn=depth_res_mpnn, num_edge_layers=num_mpnn_edge_layers) for _ in range(num_res_mpnn_layers)])
        self.readout = Readout(res_embedding_dim=res_embedding_dim, readout_hidden_dim=readout_hidden_dim, num_layers=num_readout_layers, dropout=dropout)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RNAMPNN model.

        Args:
            coords (torch.Tensor): Atom coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask indicating valid residues of shape (batch_size, max_len).

        Returns:
            torch.Tensor: Predicted residue type logits of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS).
        """
        atom_coords, atom_mask = to_atom_format(coords, mask)
        atom_embedding, atom_cross_dists, atom_edge_index = self.atom_feature(atom_coords, atom_mask)
        for layer in self.atom_mpnn_layers:
            atom_embedding = layer(atom_embedding, atom_cross_dists, atom_edge_index, atom_mask)
        res_embedding, res_edge_embedding, edge_index = self.res_feature(coords, mask, atom_embedding)
        for layer in self.res_mpnn_layers:
            res_embedding, res_edge_embedding = layer(res_embedding, res_edge_embedding, edge_index, mask)
        logits = self.readout(res_embedding, mask)

        return logits

    def training_step(self, batch):
        sequences, coords, mask, _ = batch
        sequences.to(self.device)
        coords = coords.to(self.device)
        mask = mask.to(self.device)
        logits = self(coords, mask)
        print(sequences.view(-1).shape)
        print(F.softmax(logits, dim=-1).view(-1).shape)
        loss = self.loss_fn(F.softmax(logits, dim=-1).view(-1), sequences.view(-1))

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        """
        Perform a validation step, computing loss and sequence recovery rate.

        Args:
            batch: A batch of validation data.

        Returns:
            dict: Validation metrics including loss and recovery rate.
        """
        sequences, coords, mask, _ = batch
        sequences = sequences.to(self.device)
        coords = coords.to(self.device)
        mask = mask.to(self.device)

        # Forward pass
        logits = self(coords, mask)

        # Compute loss
        loss = self.loss_fn(F.softmax(logits, dim=-1).view(-1), sequences.view(-1))
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        # Compute sequence recovery rate
        probs = F.softmax(logits, dim=-1)
        correct = (probs.argmax(dim=-1) == sequences.argmax(dim=-1)) * mask  # Mask padding residues
        recovery_rate = correct.sum().item() / mask.sum().item()
        self.log('val_recovery_rate', recovery_rate, prog_bar=True, sync_dist=True)

        return {'val_loss': loss, 'val_recovery_rate': recovery_rate}

    def test_step(self, batch):
        _, coords, mask, pdb_id = batch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]