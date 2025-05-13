from typing import Tuple
from torch import nn
import torch
from .functional import GraphNormalization


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


class ResMPNN(nn.Module):
    def __init__(self,
                 res_embedding_dim: int,
                 res_edge_embedding_dim: int,
                 depth_res_mpnn: int,
                 num_edge_layers: int,
                 dropout: float):
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
        res_edge_embedding = self.edge_layers(concatenated_features)  # Shape: (batch_size, max_len, num_neighbours, res_edge_embedding_dim)

        return res_edge_embedding

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
