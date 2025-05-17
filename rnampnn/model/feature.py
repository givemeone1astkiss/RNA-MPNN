from torch import nn
import torch
from ..config.glob import DEFAULT_HIDDEN_DIM, NUM_MAIN_SEQ_ATOMS, LEPS, SEPS
from typing import Tuple
from torch.nn import functional as F
from .functional import GraphNormalization, RNABert


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
    atom_coords = coords.view(coords.shape[0], -1, coords.shape[3])
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


class ResFeature(nn.Module):
    def __init__(self,
                 num_neighbours: int,
                 num_inside_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_inside_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_inside_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS-1,
                 num_cross_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_cross_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 res_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 padding_len: int = 4500,
                 num_attn_layers: int = 2,
                 num_heads=4,
                 ffn_dim=DEFAULT_HIDDEN_DIM,
                 num_ffn_layers=2,
                 res_edge_embedding_dim: int = DEFAULT_HIDDEN_DIM,
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
            res_embedding_dim (int): Dimension of the residue-level node features.
            res_edge_embedding_dim (int): Dimension of the residue-level edge features.
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
        raw_dim = num_inside_dist_atoms * (num_inside_dist_atoms - 1) // 2 + num_inside_angle_atoms - 2 + num_inside_dihedral_atoms - 3
        raw_edge_dim = num_cross_dist_atoms ** 2 + (num_cross_angle_atoms - 1) ** 2 + (num_cross_dihedral_atoms - 2) ** 2

        self.raw_project = nn.Linear(raw_dim, res_embedding_dim)
        self.res_embedding = RNABert(padding_len=padding_len,
                                       res_embedding_dim=res_embedding_dim,
                                       num_attn_layers=num_attn_layers,
                                       num_heads=num_heads,
                                       ffn_dim=ffn_dim,
                                       num_ffn_layers=num_ffn_layers,
                                       dropout=dropout)

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
        batch_size, max_len, _, _ = coords.shape

        avg_coords = coords.mean(dim=2)  # Shape: (batch_size, max_len, 3)

        d_coords = avg_coords.unsqueeze(1) - avg_coords.unsqueeze(2)  # Shape: (batch_size, max_len, max_len, 3)
        distances = torch.sqrt(torch.sum(d_coords ** 2, dim=-1) + SEPS)  # Shape: (batch_size, max_len, max_len)

        residue_mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # Shape: (batch_size, max_len, max_len)
        distances = distances * residue_mask_2d + (1.0 - residue_mask_2d) * LEPS

        diagonal_mask = torch.eye(max_len, device=distances.device).unsqueeze(0)  # Shape: (1, max_len, max_len)
        distances = distances + diagonal_mask * LEPS

        _, edge_index = torch.topk(
            distances,
            min(self.num_neighbours, max_len),  # Ensure we don't request more neighbors than available
            dim=-1,
            largest=False
        )

        if self.num_neighbours > max_len:
            padding_size = self.num_neighbours - max_len
            edge_index = torch.cat(
                [edge_index,
                torch.full((batch_size, max_len, padding_size), -1, device=edge_index.device, dtype=edge_index.dtype)],
                dim=-1
            )

        indices = torch.arange(max_len, device=edge_index.device).view(1, -1, 1)  # Shape: (1, max_len, 1)

        edge_index = torch.where(edge_index == indices, -1, edge_index)

        valid_neighbors = (residue_mask_2d.sum(dim=-1) - 1).clamp(min=0)
        padding_mask = valid_neighbors.unsqueeze(-1) < torch.arange(self.num_neighbours, device=distances.device)

        edge_index = edge_index.masked_fill(padding_mask, -1)

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
            inside_dists (torch.Tensor): Pairwise distances of shape (batch_size, max_len, num_inside_dist_atoms * (num_inside_dist_atoms - 1) / 2).
        """
        batch_size, max_len, num_atoms, _ = coords.shape
        assert num_atoms >= self.num_inside_dist_atoms, f"NUM_MAIN_SEQ_ATOMS({num_atoms}) must be at least num_inside_dist_atoms({self.num_inside_dist_atoms}) for distance calculation."

        # Truncate to the first `num_inside_dist_atoms`
        coords = coords[:, :, :self.num_inside_dist_atoms, :]  # Shape: (batch_size, max_len, num_inside_dist_atoms, 3)

        # Compute pairwise distances
        diffs = coords.unsqueeze(3) - coords.unsqueeze(
            2)  # Shape: (batch_size, max_len, num_inside_dist_atoms, num_inside_dist_atoms, 3)
        pairwise_dists = torch.sqrt(torch.sum(diffs ** 2,
                                              dim=-1) + SEPS)  # Shape: (batch_size, max_len, num_inside_dist_atoms, num_inside_dist_atoms)

        # Extract upper triangular part (excluding diagonal)
        triu_indices = torch.triu_indices(self.num_inside_dist_atoms, self.num_inside_dist_atoms, offset=1,
                                          device=coords.device)
        inside_dists = pairwise_dists[:, :, triu_indices[0], triu_indices[
                                                                 1]]  # Shape: (batch_size, max_len, num_inside_dist_atoms * (num_inside_dist_atoms - 1) / 2)

        # Mask invalid distances for padding residues
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

        coords = coords[:, :, :self.num_inside_angle_atoms, :]  # Shape: (batch_size, max_len, num_inside_angle_atoms, 3)

        vecs = coords[:, :, 1:] - coords[:, :, :-1]  # Shape: (batch_size, max_len, num_inside_angle_atoms-1, 3)

        dot_products = (vecs[:, :, :-1] * vecs[:, :, 1:]).sum(dim=-1)  # Shape: (batch_size, max_len, num_inside_angle_atoms-2)

        norms = torch.norm(vecs, dim=-1)  # Shape: (batch_size, max_len, num_inside_angle_atoms-1)

        inside_angles = dot_products / (norms[:, :, :-1] * norms[:, :, 1:] + SEPS)  # Shape: (batch_size, max_len, num_inside_angle_atoms-2)

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

        coords = coords[:, :, :self.num_inside_dihedral_atoms, :]  # Shape: (batch_size, max_len, num_inside_dihedral_atoms, 3)

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

    def _res_embedding(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute residue-level graph node features.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).

        Returns:
            res_embedding (torch.Tensor): Residue embedding of shape (batch_size, max_len, res_embedding_dim).
        """
        inside_dists = self._inside_dists(coords, mask)  # Shape: (batch_size, max_len, num_inside_dist_atoms * (num_inside_dist_atoms - 1) / 2)
        inside_angles = self._inside_angles(coords, mask)  # Shape: (batch_size, max_len, num_inside_angle_atoms - 2)
        inside_dihedrals = self._inside_dihedrals(coords, mask)  # Shape: (batch_size, max_len, num_inside_dihedral_atoms - 3)

        raw = torch.cat([inside_dists, inside_angles, inside_dihedrals], dim=-1)  # Shape: (batch_size, max_len, raw_dim)
        res_embedding = self.res_embedding(self.raw_project(raw), mask)  # Shape: (batch_size, max_len, res_embedding_dim)

        return res_embedding

    def _res_edge_embedding(self, coords: torch.Tensor, mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute residue-level edge features.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).
            edge_index (torch.Tensor): Indices of neighbors of shape (batch_size, max_len, num_neighbours).

        Returns:
            res_edge_embedding (torch.Tensor): Residue edge embedding of shape (batch_size, max_len, num_neighbours, res_edge_embedding_dim).
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

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method to compute residue-level graph embeddings.

        Args:
            coords (torch.Tensor): Residue-level coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Residue-level mask of shape (batch_size, max_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - res_embedding: Residue embedding of shape (batch_size, max_len, res_embedding_dim).
                - res_edge_embedding: Residue edge embedding of shape (batch_size, max_len, num_neighbours, res_edge_embedding_dim).
                - edge_index: Indices of neighbors of shape (batch_size, max_len, num_neighbours).
        """
        edge_index = self._get_res_graph(coords, mask)  # Shape: (batch_size, max_len, num_neighbours)

        res_edge_embedding = self._res_edge_embedding(coords, mask, edge_index)  # Shape: (batch_size, max_len, num_neighbours, res_edge_embedding_dim)

        res_embedding = self._res_embedding(coords, mask)  # Shape: (batch_size, max_len, res_embedding_dim)
        res_embedding = self.graph_norm(res_embedding, mask)
        return res_embedding, res_edge_embedding, edge_index