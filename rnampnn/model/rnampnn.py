import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from typing import Union, List, Tuple

class RNAFeatures(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

class RNAMPNN(LightningModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def predict(self, *args, **kwargs):
        pass

class AtomGraph(Data):
    def __init__(self, rna_array: np.ndarray, k: int) -> None:
        """
        Initialize the RNAData class.

        Args:
            rna_array (np.ndarray): A (len, 7, 3) array representing the RNA sequence.
                Each unit contains 7 atoms, and each atom has 3 Euclidean coordinates.
            k (int): The number of top-k nearest neighbors to consider for each atom.

        """
        # Convert the input RNA array to a PyTorch tensor
        rna_tensor = torch.tensor(rna_array, dtype=torch.float32)
        flattened_coords = rna_tensor.view(-1, 3)

        # Compute the top-k nearest neighbors and edge indices
        edge_index = knn_graph(flattened_coords, k=k, batch=None, loop=False)

        # Add self-loops to the edge indices
        edge_index, _ = add_self_loops(edge_index, num_nodes=flattened_coords.size(0))

        # Compute pairwise distances
        row, col = edge_index
        distances = torch.norm(flattened_coords[row] - flattened_coords[col], dim=1).unsqueeze(1)

        # Assign one-hot labels to each atom as initial features
        num_atom_types = 7
        atom_types = torch.arange(num_atom_types).repeat(rna_tensor.shape[0])
        x = torch.nn.functional.one_hot(atom_types, num_classes=num_atom_types).float()

        # Initialize the Data object with features, edge indices, positions, and edge attributes
        super().__init__(x=x, edge_index=edge_index, pos=flattened_coords, edge_attr=distances)

class ResGraph(Data):
    def __init__(self, rna_array: Union[torch.Tensor, list], residue_features: torch.Tensor, k: int) -> None:
        """
        Initialize the ResGraph class.

        Args:
            rna_array (Union[torch.Tensor, list]): A (len, 7, 3) array representing the RNA sequence.
                Each unit contains 7 atoms, and each atom has 3 Euclidean coordinates.
            residue_features (torch.Tensor): A (len, dim) tensor representing residue-level features.
            k (int): The number of top-k nearest neighbors to consider for each residue.
        """
        # Convert the input RNA array to a PyTorch tensor
        rna_tensor = torch.tensor(rna_array, dtype=torch.float32)  # Shape: (len, 7, 3)
        num_residues = rna_tensor.shape[0]

        # Compute residue coordinates as the mean of the 7 atoms
        residue_coords = rna_tensor.mean(dim=1)  # Shape: (len, 3)

        # Compute plane and dihedral angles
        plane_cosines = self._compute_plane_angles(rna_tensor)
        dihedral_cosines = self._compute_dihedral_angles(rna_tensor)

        # Concatenate plane and dihedral cosines
        angle_features = torch.cat([plane_cosines, dihedral_cosines], dim=1)  # Shape: (len, 8)

        # Concatenate angle features with residue-level features
        node_features = torch.cat([angle_features, residue_features], dim=1)  # Shape: (len, 8 + dim)

        # Compute top-k nearest neighbors using residue coordinates
        edge_index = knn_graph(residue_coords, k=k, batch=None, loop=False)  # Shape: (2, num_edges)

        # Initialize the Data object with features, edge indices, and positions
        super().__init__(x=node_features, edge_index=edge_index, pos=residue_coords)

    @staticmethod
    def _compute_plane_angles(rna_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine values of the plane angles for each residue.

        Args:
            rna_tensor (torch.Tensor): A (len, 7, 3) tensor representing the RNA sequence.

        Returns:
            torch.Tensor: A (len, 3) tensor containing the cosine values of the plane angles.
        """
        planes = rna_tensor[:, :6, :]
        vectors = planes[:, 1:, :] - planes[:, :-1, :]
        normals = torch.linalg.cross(vectors[:, :-1, :], vectors[:, 1:, :], dim=2)
        plane_cosines = torch.nn.functional.cosine_similarity(normals[:, :-1, :], normals[:, 1:, :], dim=2)
        return plane_cosines

    @staticmethod
    def _compute_dihedral_angles(rna_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine values of the dihedral angles for each residue.

        Args:
            rna_tensor (torch.Tensor): A (len, 7, 3) tensor representing the RNA sequence.

        Returns:
            torch.Tensor: A (len, 4) tensor containing the cosine values of the dihedral angles.
        """
        planes = rna_tensor[:, :6, :]
        vectors = planes[:, 1:, :] - planes[:, :-1, :]
        dihedral_cosines = torch.nn.functional.cosine_similarity(vectors[:, :-2, :], vectors[:, 2:, :], dim=2)
        return dihedral_cosines
