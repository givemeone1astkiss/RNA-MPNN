import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from typing import Tuple
from torch import Tensor
from ..config.glob import NUM_MAIN_SEQ_ATOMS
from torch.nn import functional as F


class GraphNormalization(nn.Module):
    def __init__(self, embedding_dim: int, epsilon: float = 1e-5):
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
    def __init__(self, num_atom_neighbour: int, atom_embedding_dim: int = 32):
        super().__init__()
        self.num_atom_neighbour = num_atom_neighbour
        self.atom_embedding_dim = atom_embedding_dim
        self.embedding = nn.Embedding(num_embeddings=NUM_MAIN_SEQ_ATOMS, embedding_dim=self.atom_embedding_dim)
        self.normalization = GraphNormalization(embedding_dim=self.atom_embedding_dim)

    def _get_atom_graph(self, atom_coords: torch.Tensor, atom_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a k-nearest neighbor graph for RNA atom coordinates.

        Args:
            atom_coords (torch.Tensor): Atom coordinates of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, 3).
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - atom_cross_dists: Distances to neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).
                - atom_edge_index: Indices of neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).
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
        atom_cross_dists, atom_edge_idx = torch.topk(distances, min(self.num_atom_neighbour, num_atoms), dim=-1, largest=False)

        # Handle cases where num_atom_neighbour exceeds the number of valid atoms
        if self.num_atom_neighbour > num_atoms:
            padding_size = self.num_atom_neighbour - num_atoms
            atom_cross_dists = torch.cat(
                [atom_cross_dists, torch.full((batch_size, num_atoms, padding_size), 1e6, device=distances.device)],
                dim=-1
            )
            atom_edge_idx = torch.cat(
                [atom_edge_idx,
                 torch.full((batch_size, num_atoms, padding_size), -1, device=distances.device, dtype=torch.long)],
                dim=-1
            )

        # Set all neighbors of padding atoms to -1 and distances to 1e6
        padding_atom_mask = (atom_mask == 0).unsqueeze(-1)  # (batch_size, num_atoms, 1)
        atom_edge_idx = atom_edge_idx.masked_fill(padding_atom_mask, -1)
        atom_cross_dists = atom_cross_dists.masked_fill(padding_atom_mask, 1e6)

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
        return self.normalization(embeddings, atom_mask)

    def forward(self, atom_coords: torch.Tensor, atom_mask: torch.Tensor):
        """
        Forward pass to process the coordinates and mask.
        Args:
            atom_coords (torch.Tensor): Atom coordinates of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, 3).
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            atom_embedding: Encoded atom features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
            atom_cross_dists: Distances to neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).
            atom_edge_index: Indices of neighbors (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).
        """
        atom_embedding = self._atom_embedding(atom_mask)
        atom_cross_dists, atom_edge_index = self._get_atom_graph(atom_coords, atom_mask)
        return atom_embedding, atom_cross_dists, atom_edge_index


class AtomMPNN(nn.Module):
    def __init__(self, atom_hidden_dim: int, num_layers: int, dropout: float = 0.1):
        """
        Message Passing Neural Network (MPNN) for atom-level features.
        Args:
            atom_hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of linear layers for message generation.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.atom_hidden_dim = atom_hidden_dim
        self.num_layers = num_layers
        self.graph_norm = GraphNormalization(embedding_dim=atom_hidden_dim)

        layers = []
        input_dim = atom_hidden_dim * 2 + 1
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, atom_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = atom_hidden_dim
        self.message_layers = nn.Sequential(*layers)

    def message(self, atom_mask: torch.Tensor, atom_embedding: torch.Tensor, atom_cross_dists: torch.Tensor,
                atom_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Generate messages for each edge in the graph.

        Args:
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            atom_embedding (torch.Tensor): Node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_hidden_dim).
            atom_cross_dists (torch.Tensor): Distances to neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).
            atom_edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).

        Returns:
            torch.Tensor: Messages of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour, atom_hidden_dim).
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
            safe_edge_index.unsqueeze(-1).expand(-1, -1, -1, self.atom_hidden_dim)
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
    def aggregation(atom_mask: torch.Tensor, atom_embedding: torch.Tensor, messages: torch.Tensor, atom_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages for each node and update node features.

        Args:
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            atom_embedding (torch.Tensor): Node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_hidden_dim).
            messages (torch.Tensor): Messages of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour, atom_hidden_dim).
            atom_edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).

        Returns:
            torch.Tensor: Updated node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_hidden_dim).
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

    def forward(self, atom_embedding: torch.Tensor, atom_mask: torch.Tensor, atom_cross_dists: torch.Tensor, atom_edge_index: torch.Tensor) -> \
    tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the AtomMPNN module.

        Args:
            atom_mask (torch.Tensor): Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
            atom_embedding (torch.Tensor): Node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_hidden_dim).
            atom_cross_dists (torch.Tensor): Distances to neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).
            atom_edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - atom_embedding: Updated node features of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_hidden_dim).
                - atom_mask: Mask indicating valid atoms of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).
                - atom_cross_dists: Distances to neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).
                - atom_edge_index: Indices of neighbors of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, num_atom_neighbour).
        """
        messages = self.message(atom_mask, atom_embedding, atom_cross_dists, atom_edge_index)
        atom_embedding = self.aggregation(atom_mask, atom_embedding, messages, atom_edge_index)
        atom_embedding = self.graph_norm(atom_embedding, atom_mask)
        return atom_embedding, atom_mask, atom_cross_dists, atom_edge_index


class AtomPooling(nn.Module):
    def __init__(self, raw_dim: int, atom_hidden_dim: int, num_layers: int, dropout: float):
        """
        Args:
            raw_dim (int): Dimension of the raw residue-level features.
            atom_hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of linear layers for computing weights.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.num_layers = num_layers
        self.atom_hidden_dim = atom_hidden_dim

        layers = []
        input_dim = raw_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, atom_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = atom_hidden_dim
        layers.append(nn.Linear(atom_hidden_dim, NUM_MAIN_SEQ_ATOMS))
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
                 num_neighbour: int,
                 num_inside_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_inside_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_inside_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS-1,
                 num_cross_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_cross_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 atom_pooling_hidden_dim: int = 32,
                 atom_embedding_dim: int = 32,
                 res_embedding_dim: int = 32,
                 res_edge_embedding_dim: int = 32,
                 num_atom_pooling_layers: int = 2,
                 num_layers: int = 2,
                 num_edge_layers: int = 2,
                 dropout: float = 0.1):
        """
        Residue-level feature extraction module.
        Args:
            num_neighbour (int): Number of neighbors for each residue.
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
        self.num_neighbour = num_neighbour
        assert num_inside_dist_atoms >= 2, "num_inside_dist_atoms must be at least 2 for distance calculation."
        self.num_inside_dist_atoms = num_inside_dist_atoms
        assert num_inside_angle_atoms >= 3, "num_inside_angle_atoms must be at least 3 for angle calculation."
        self.num_inside_angle_atoms = num_inside_angle_atoms
        assert num_inside_dihedral_atoms >= 4, "num_inside_dihedral_atoms must be at least 4 for dihedral calculation."
        self.num_inside_dihedral_atoms = num_inside_dihedral_atoms
        assert num_cross_dist_atoms >= 2, "num_cross_dist_atoms must be at least 2 for cross distance calculation."
        self.num_cross_dist_atoms = num_cross_dist_atoms
        assert num_cross_angle_atoms >= 3, "num_cross_angle_atoms must be at least 3 for cross angle calculation."
        self.num_cross_angle_atoms = num_cross_angle_atoms
        assert num_cross_dihedral_atoms >= 4, "num_cross_dihedral_atoms must be at least 4 for cross dihedral calculation."
        self.num_cross_dihedral_atoms = num_cross_dihedral_atoms
        raw_dim = num_inside_dist_atoms - 1 + num_inside_angle_atoms - 2 + num_inside_dihedral_atoms - 3
        raw_edge_dim = (num_cross_dist_atoms * (num_cross_dist_atoms - 1) // 2 +
                             (num_cross_angle_atoms - 1) * (num_cross_angle_atoms - 2) // 2 +
                             (num_cross_dihedral_atoms - 2) * (num_cross_dihedral_atoms - 3) // 2)
        self.atom_pooling = AtomPooling(raw_dim=raw_dim,
                                        atom_hidden_dim=atom_pooling_hidden_dim,
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

        layers = []
        input_dim = raw_edge_dim
        for _ in range(num_edge_layers):
            layers.append(nn.Linear(input_dim, res_edge_embedding_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = res_edge_embedding_dim
        self.res_edge_embedding_layers = nn.Sequential(*layers)

    def _get_res_graph(self, coords: torch.Tensor, mask: torch.Tensor) -> Tensor:
        """
        Generate residue-level graph structure.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).

        Returns:
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len, num_neighbour).
        """
        batch_size, max_len, num_atoms, _ = coords.shape

        # Compute average coordinates for each residue
        avg_coords = coords.mean(dim=2)  # Shape: (batch_size, max_len, 3)

        # Compute pairwise distances between residues
        d_coords = avg_coords.unsqueeze(1) - avg_coords.unsqueeze(2)  # Shape: (batch_size, max_len, max_len, 3)
        distances = torch.sqrt(torch.sum(d_coords ** 2, dim=-1) + 1e-6)  # Shape: (batch_size, max_len, max_len)

        # Mask invalid distances (real residues to padding residues)
        residue_mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # Shape: (batch_size, max_len, max_len)
        distances = distances * residue_mask_2d + (1.0 - residue_mask_2d) * 1e6

        # Select k-nearest neighbors
        dist_neighbour, edge_index = torch.topk(distances, min(self.num_neighbour, max_len), dim=-1, largest=False)

        # Handle cases where num_neighbour exceeds the number of valid residues
        if self.num_neighbour > max_len:
            padding_size = self.num_neighbour - max_len
            dist_neighbour = torch.cat(
                [dist_neighbour, torch.full((batch_size, max_len, padding_size), 1e6, device=distances.device)],
                dim=-1
            )
            edge_index = torch.cat(
                [edge_index,
                 torch.full((batch_size, max_len, padding_size), -1, device=distances.device, dtype=torch.long)],
                dim=-1
            )

        # Set all neighbors of padding residues to -1 and distances to 1e6
        padding_residue_mask = (mask == 0).unsqueeze(-1)  # (batch_size, max_len, 1)
        edge_index = edge_index.masked_fill(padding_residue_mask, -1)

        return edge_index

    def _inside_dists(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between all atoms within each sequence unit.

        Args:
            coords (torch.Tensor): Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask of shape (batch_size, max_len).

        Returns:
            inside_dists (torch.Tensor): Pairwise distances of shape (batch_size, max_len, num_inside_dist_atoms * (num_inside_dist_atoms - 1) / 2).
        """
        batch_size, max_len, num_atoms, _ = coords.shape
        assert num_atoms >= self.num_inside_dist_atoms, "NUM_MAIN_SEQ_ATOMS must be at least num_inside_dist_atoms for distance calculation."

        coords = coords[:, :, :self.num_inside_dist_atoms, :]
        # Compute pairwise distances between all atoms
        d_coords = coords.unsqueeze(-2) - coords.unsqueeze(-3)  # Shape: (batch_size, max_len, num_atoms, num_atoms, 3)
        distances = torch.sqrt(torch.sum(d_coords ** 2, dim=-1) + 1e-6)  # Shape: (batch_size, max_len, num_atoms, num_atoms)

        # Extract the upper triangular part of the distance matrix (excluding the diagonal)
        triu_indices = torch.triu_indices(num_atoms, num_atoms, offset=1, device=coords.device)
        inside_dists = distances[:, :, triu_indices[0], triu_indices[1]]  # Shape: (batch_size, max_len, num_atoms * (num_atoms - 1) / 2)

        # Mask invalid residues (set distances to zero for padding residues)
        inside_dists = inside_dists * mask.unsqueeze(-1)

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
        batch_size, max_len, num_atoms, _ = coords.shape
        assert num_atoms >= self.num_inside_angle_atoms, "NUM_MAIN_SEQ_ATOMS must be at least num_inside_angle_atoms for angle calculation."

        # Truncate to the first six atoms
        coords = coords[:, :, :self.num_inside_angle_atoms, :]  # Shape: (batch_size, max_len, num_inside_angle_atoms, 3)

        # Compute vectors between adjacent atoms
        vecs = coords[:, :, 1:] - coords[:, :, :-1]  # Shape: (batch_size, max_len, num_inside_angle_atoms-1, 3)

        # Compute dot products between consecutive vectors
        dot_products = (vecs[:, :, :-1] * vecs[:, :, 1:]).sum(dim=-1)  # Shape: (batch_size, max_len, num_inside_angle_atoms-2)

        # Compute norms of the vectors
        norms = torch.norm(vecs, dim=-1)  # Shape: (batch_size, max_len, num_inside_angle_atoms-1)

        # Compute cosine values for the angles
        inside_angles = dot_products / (norms[:, :, :-1] * norms[:, :, 1:] + 1e-6)  # Shape: (batch_size, max_len, num_inside_angle_atoms-2)

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
            inside_dihedral_angles (torch.Tensor): Dihedral angles of shape (batch_size, max_len, num_inside_dihedral_atoms-3).
        """
        batch_size, max_len, num_atoms, _ = coords.shape
        assert num_atoms >= self.num_inside_dihedral_atoms, "NUM_MAIN_SEQ_ATOMS must be at least num_inside_dihedral_atoms for dihedral calculation."

        # Truncate to the first num_inside_dihedral_atoms
        coords = coords[:, :, :self.num_inside_dihedral_atoms, :]  # Shape: (batch_size, max_len, num_inside_dihedral_atoms, 3)

        # Compute vectors between consecutive atoms
        vec1 = coords[:, :, 1:] - coords[:, :, :-1]  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-1, 3)
        vec2 = vec1[:, :, 1:]  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-2, 3)
        vec1 = vec1[:, :, :-1]  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-2, 3)

        # Compute normal vectors for planes
        normal1 = torch.cross(vec1, vec2, dim=-1)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-2, 3)
        normal2 = torch.cross(vec2, vec1[:, :, 1:], dim=-1)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-3, 3)

        # Normalize the normal vectors
        normal1 = normal1 / (torch.norm(normal1, dim=-1, keepdim=True) + 1e-6)
        normal2 = normal2 / (torch.norm(normal2, dim=-1, keepdim=True) + 1e-6)

        # Compute cosine and sine of dihedral angles
        cos_angle = (normal1[:, :, 1:] * normal2).sum(dim=-1)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-3)
        sin_angle = torch.cross(normal1[:, :, 1:], normal2,dim=-1)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-3)
        sin_angle = (sin_angle * vec2[:, :, 1:]).sum(dim=-1)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-3)

        # Compute dihedral angles using atan2
        inside_dihedral_angles = torch.atan2(sin_angle, cos_angle)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms-3)

        # Mask invalid residues (set dihedral angles to zero for padding residues)
        inside_dihedral_angles = inside_dihedral_angles * mask.unsqueeze(-1)

        return inside_dihedral_angles

    def _cross_dists(self, coords: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between atoms of a residue and its neighbors.

        Args:
            coords (torch.Tensor): Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask of shape (batch_size, max_len).
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len, num_neighbour).

        Returns:
            inside_cross_dists (torch.Tensor): Cross distances of shape (batch_size, max_len, num_neighbour, num_cross_dist_atoms * (num_cross_dist_atoms - 1) / 2).
        """
        assert coords.shape[2] >= self.num_cross_dist_atoms, "NUM_MAIN_SEQ_ATOMS must be at least num_cross_dist_atoms for cross distance calculation."
        coords = coords[:, :, :self.num_cross_dist_atoms, :]  # Shape: (batch_size, max_len, num_cross_dist_atoms, 3)

        batch_size, max_len, num_atoms, _ = coords.shape

        num_neighbour = edge_index.shape[-1]

        # Gather neighbor coordinates using edge_index
        safe_edge_index = edge_index.clone()
        safe_edge_index[safe_edge_index == -1] = 0  # Replace invalid indices with 0
        neighbor_coords = torch.gather(
            coords.unsqueeze(2).expand(-1, -1, num_neighbour, -1, -1),
            1,
            safe_edge_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, num_atoms, 3)
        )  # Shape: (batch_size, max_len, num_neighbour, NUM_MAIN_SEQ_ATOMS, 3)

        # Compute pairwise distances between atoms of the residue and its neighbors
        d_coords = coords.unsqueeze(2).unsqueeze(-2) - neighbor_coords.unsqueeze(
            -3)  # Shape: (batch_size, max_len, num_neighbour, NUM_MAIN_SEQ_ATOMS, NUM_MAIN_SEQ_ATOMS, 3)
        distances = torch.sqrt(torch.sum(d_coords ** 2,dim=-1) + 1e-6)  # Shape: (batch_size, max_len, num_neighbour, NUM_MAIN_SEQ_ATOMS, NUM_MAIN_SEQ_ATOMS)

        # Extract the upper triangular part of the distance matrix (excluding the diagonal)
        triu_indices = torch.triu_indices(num_atoms, num_atoms, offset=1, device=coords.device)
        cross_dists = distances[:, :, :, triu_indices[0], triu_indices[1]]  # Shape: (batch_size, max_len, num_neighbour, NUM_MAIN_SEQ_ATOMS * (NUM_MAIN_SEQ_ATOMS - 1) / 2)

        residue_mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, max_len, 1, 1)
        neighbor_mask = (edge_index != -1).unsqueeze(-1)  # Shape: (batch_size, max_len, num_neighbour, 1)

        invalid_mask = (residue_mask * neighbor_mask) == 0
        inside_cross_dists = cross_dists.masked_fill(invalid_mask, 1e6)

        return inside_cross_dists

    def _cross_angles(self, coords: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine of angles formed by every two consecutive vectors in each residue and its neighbors.

        Args:
            coords (torch.Tensor): Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask of shape (batch_size, max_len).
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len, num_neighbour).

        Returns:
            cross_angles (torch.Tensor): Cosine of angles of shape (batch_size, max_len, num_neighbour, (num_cross_dist_atoms - 1) * (num_cross_dist_atoms - 2) / 2).
        """
        assert coords.shape[2] >= self.num_cross_angle_atoms, "NUM_MAIN_SEQ_ATOMS must be at least num_cross_angle_atoms for angle calculation."
        coords = coords[:, :, :self.num_cross_angle_atoms, :]  # Truncate to the first num_cross_angle_atoms

        batch_size, max_len, num_atoms, _ = coords.shape
        num_neighbour = edge_index.shape[-1]

        # Gather neighbor coordinates using edge_index
        safe_edge_index = edge_index.clone()
        safe_edge_index[safe_edge_index == -1] = 0  # Replace invalid indices with 0
        neighbor_coords = torch.gather(
            coords.unsqueeze(2).expand(-1, -1, num_neighbour, -1, -1),
            1,
            safe_edge_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, num_atoms, 3)
        )  # Shape: (batch_size, max_len, num_neighbour, num_atoms, 3)

        # Compute vectors between consecutive atoms
        vectors = neighbor_coords[:, :, :, 1:] - neighbor_coords[:, :, :,
                                                 :-1]  # Shape: (batch_size, max_len, num_neighbour, num_atoms-1, 3)

        # Compute pairwise cosine of angles
        vec1 = vectors.unsqueeze(-2)  # Shape: (batch_size, max_len, num_neighbour, num_atoms-2, 1, 3)
        vec2 = vectors.unsqueeze(-3)  # Shape: (batch_size, max_len, num_neighbour, 1, num_atoms-2, 3)
        dot_product = (vec1 * vec2).sum(dim=-1)  # Shape: (batch_size, max_len, num_neighbour, num_atoms-2, num_atoms-2)
        norm1 = torch.norm(vec1, dim=-1)  # Shape: (batch_size, max_len, num_neighbour, num_atoms-2, 1)
        norm2 = torch.norm(vec2, dim=-1)  # Shape: (batch_size, max_len, num_neighbour, 1, num_atoms-2)
        cosine_angles = dot_product / (
                    norm1 * norm2 + 1e-6)  # Shape: (batch_size, max_len, num_neighbour, num_atoms-2, num_atoms-2)

        # Extract upper triangular part of the cosine matrix
        triu_indices = torch.triu_indices(num_atoms - 1, num_atoms - 1, offset=1, device=coords.device)
        cross_angles = cosine_angles[:, :, :, triu_indices[0], triu_indices[1]]  # Shape: (batch_size, max_len, num_neighbour, (num_atoms-1)*(num_atoms-2)/2)

        # Mask invalid residues and neighbors
        residue_mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, max_len, 1, 1)
        neighbor_mask = (edge_index != -1).unsqueeze(-1)  # Shape: (batch_size, max_len, num_neighbour, 1)
        valid_mask = residue_mask * neighbor_mask
        cross_angles = cross_angles * valid_mask

        return cross_angles

    def _cross_dihedrals(self, coords: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine of dihedral angles formed by every four consecutive atoms in each residue and its neighbors.

        Args:
            coords (torch.Tensor): Coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask of shape (batch_size, max_len).
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len, num_neighbour).

        Returns:
            cross_dihedral_angles (torch.Tensor): Cosine of dihedral angles of shape (batch_size, max_len, num_neighbour, (num_cross_dist_atoms - 2) * (num_cross_dist_atoms - 3) / 2).
        """
        assert coords.shape[
                   2] >= self.num_cross_dihedral_atoms, "NUM_MAIN_SEQ_ATOMS must be at least num_cross_dihedral_atoms for dihedral calculation."
        coords = coords[:, :, :self.num_cross_dihedral_atoms, :]  # Truncate to the first num_cross_dihedral_atoms

        batch_size, max_len, num_atoms, _ = coords.shape
        num_neighbour = edge_index.shape[-1]

        # Gather neighbor coordinates using edge_index
        safe_edge_index = edge_index.clone()
        safe_edge_index[safe_edge_index == -1] = 0  # Replace invalid indices with 0
        neighbor_coords = torch.gather(
            coords.unsqueeze(2).expand(-1, -1, num_neighbour, -1, -1),
            1,
            safe_edge_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, num_atoms, 3)
        )  # Shape: (batch_size, max_len, num_neighbour, num_atoms, 3)

        # Compute vectors between consecutive atoms
        vec1 = neighbor_coords[:, :, :, 1:] - neighbor_coords[:, :, :,:-1]  # Shape: (batch_size, max_len, num_neighbour, num_atoms-1, 3)
        vec2 = vec1[:, :, :, 1:]  # Shape: (batch_size, max_len, num_neighbour, num_atoms-2, 3)
        vec1 = vec1[:, :, :, :-1]  # Shape: (batch_size, max_len, num_neighbour, num_atoms-2, 3)

        # Compute normal vectors for planes
        normal1 = torch.cross(vec1, vec2, dim=-1)  # Shape: (batch_size, max_len, num_neighbour, num_atoms-2, 3)
        normal2 = torch.cross(vec2, vec1[:, :, :, 1:],dim=-1)  # Shape: (batch_size, max_len, num_neighbour, num_atoms-3, 3)

        # Normalize the normal vectors
        normal1 = normal1 / (torch.norm(normal1, dim=-1, keepdim=True) + 1e-6)
        normal2 = normal2 / (torch.norm(normal2, dim=-1, keepdim=True) + 1e-6)

        # Compute cosine of dihedral angles
        cos_angle = (normal1[:, :, :, 1:] * normal2).sum(dim=-1)  # Shape: (batch_size, max_len, num_neighbour, num_atoms-3)

        # Extract upper triangular part of the cosine matrix
        triu_indices = torch.triu_indices(num_atoms - 2, num_atoms - 2, offset=1, device=coords.device)
        cross_dihedral_angles = cos_angle[:, :, :, triu_indices[0], triu_indices[1]]  # Shape: (batch_size, max_len, num_neighbour, (num_atoms-2)*(num_atoms-3)/2)

        # Mask invalid residues and neighbors
        residue_mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, max_len, 1, 1)
        neighbor_mask = (edge_index != -1).unsqueeze(-1)  # Shape: (batch_size, max_len, num_neighbour, 1)
        valid_mask = residue_mask * neighbor_mask
        cross_dihedral_angles = cross_dihedral_angles * valid_mask

        return cross_dihedral_angles

    def _res_embedding(self, coords: torch.Tensor, mask: torch.Tensor, atom_embedding: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute residue-level graph node features.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).
            atom_embedding (torch.Tensor): Atom-level embeddings of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
            atom_mask (torch.Tensor): Atom-level mask of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            res_embedding (torch.Tensor): Residue embedding of shape (batch_size, max_len, res_embedding_dim).
        """
        inside_dists = self._inside_dists(coords, mask)  # Shape: (batch_size, max_len, num_inside_dist_atoms * (num_inside_dist_atoms - 1) / 2)
        inside_angles = self._inside_angles(coords, mask)  # Shape: (batch_size, max_len, num_inside_angle_atoms - 2)
        inside_dihedrals = self._inside_dihedrals(coords, mask)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms - 3)

        raw = torch.cat([inside_dists, inside_angles, inside_dihedrals], dim=-1)  # Shape: (batch_size, max_len, raw_dim)

        pooled_atom_embedding = self.atom_pooling(atom_embedding, atom_mask, raw)  # Shape: (batch_size, max_len, atom_hidden_dim)

        res_embedding = self.res_embedding_layers(torch.cat([raw, pooled_atom_embedding],dim=-1))  # Shape: (batch_size, max_len, res_embedding_dim)

        return res_embedding

    def _res_edge_embedding(self, coords: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute residue-level edge features.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len, num_neighbour).

        Returns:
            torch.Tensor: Residue edge embedding of shape (batch_size, max_len, num_neighbour, res_edge_embedding_dim).
        """
        cross_dists = self._cross_dists(coords, mask, edge_index)  # Shape: (batch_size, max_len, num_neighbour, num_cross_dist_atoms * (num_cross_dist_atoms - 1) / 2)
        cross_angles = self._cross_angles(coords, mask, edge_index)  # Shape: (batch_size, max_len, num_neighbour, (num_cross_angle_atoms - 1) * (num_cross_angle_atoms - 2) / 2)
        cross_dihedrals = self._cross_dihedrals(coords, mask, edge_index)  # Shape: (batch_size, max_len, num_neighbour, (num_cross_dihedral_atoms - 2) * (num_cross_dihedral_atoms - 3) / 2)

        raw_edge_features = torch.cat([cross_dists, cross_angles, cross_dihedrals], dim=-1)  # Shape: (batch_size, max_len, num_neighbour, raw_edge_dim)

        res_edge_embedding = self.res_edge_embedding_layers(raw_edge_features)  # Shape: (batch_size, max_len, num_neighbour, res_edge_embedding_dim)

        return res_edge_embedding

    def forward(self, coords: torch.Tensor, mask: torch.Tensor, atom_embedding: torch.Tensor,
                atom_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method to compute residue-level graph embeddings.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).
            atom_embedding (torch.Tensor): Atom-level embeddings of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS, atom_embedding_dim).
            atom_mask (torch.Tensor): Atom-level mask of shape (batch_size, max_len * NUM_MAIN_SEQ_ATOMS).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Residue node embeddings, edge embeddings, and edge indices.
        """
        # Generate residue graph edge information
        edge_index = self._get_res_graph(coords, mask)  # Shape: (batch_size, max_len, num_neighbour)

        # Compute residue node embeddings
        res_embedding = self._res_embedding(coords, mask, atom_embedding, atom_mask)  # Shape: (batch_size, max_len, res_embedding_dim)

        # Compute residue edge embeddings
        res_edge_embedding = self._res_edge_embedding(coords, mask, edge_index)  # Shape: (batch_size, max_len, num_neighbour, res_edge_embedding_dim)

        return res_embedding, res_edge_embedding, edge_index


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
    def __init__(self, num_atom_neighbour: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.num_atom_neighbour = num_atom_neighbour

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