import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import os
import csv
from ..config.glob import NUM_MAIN_SEQ_ATOMS, DEFAULT_HIDDEN_DIM, REVERSE_VOCAB
from torch.nn import functional as F
from .mpnn import AtomMPNN, ResMPNN
from .feature import AtomFeature, ResFeature, to_atom_format
from .utils import Readout

class RNAMPNN(LightningModule):
    def __init__(self,
                 num_atom_neighbours: int = 3,
                 atom_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 depth_atom_mpnn: int = 2,
                 num_atom_mpnn_layers = 1,
                 num_res_neighbours: int = 14,
                 num_inside_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_inside_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_inside_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_cross_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 atom_pool_hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 res_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 res_edge_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 num_atom_pool_layers: int = 2,
                 depth_res_feature: int = 2,
                 depth_res_edge_feature: int = 2,
                 num_res_mpnn_layers: int = 3,
                 depth_res_mpnn: int = 2,
                 num_mpnn_edge_layers: int = 2,
                 readout_hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 num_readout_layers: int = 3,
                 dropout: float = 0.1,
                 lr: float = 2e-3,):
        """
        Initialize the RNAMPNN model.

        Args:
            num_atom_neighbours (int): Number of neighboring atoms for atom-level features.
            atom_embedding_dim (int): Dimension of the atom embedding.
            depth_atom_mpnn (int): Depth of the atom-level MPNN.
            num_atom_mpnn_layers (int): Number of atom MPNN layers.
            num_res_neighbours (int): Number of neighboring residues for residue-level features.
            num_inside_dist_atoms (int): Number of atoms for inside distance calculation.
            num_inside_angle_atoms (int): Number of atoms for inside angle calculation.
            num_inside_dihedral_atoms (int): Number of atoms for inside dihedral calculation.
            num_cross_dist_atoms (int): Number of atoms for cross distance calculation.
            num_cross_angle_atoms (int): Number of atoms for cross angle calculation.
            num_cross_dihedral_atoms (int): Number of atoms for cross dihedral calculation.
            atom_pool_hidden_dim (int): Hidden dimension for atom Pool layers.
            res_embedding_dim (int): Dimension of the residue embedding.
            res_edge_embedding_dim (int): Dimension of the residue edge embedding.
            num_atom_pool_layers (int): Number of atom Pool layers.
            depth_res_feature (int): Depth of the residue feature extraction network.
            depth_res_edge_feature (int): Depth of the residue edge feature extraction network.
            num_res_mpnn_layers (int): Number of residue MPNN layers.
            depth_res_mpnn (int): Depth of the residue-level MPNN.
            num_mpnn_edge_layers (int): Number of edge update layers in MPNN.
            readout_hidden_dim (int): Hidden dimension for the readout layer.
            num_readout_layers (int): Number of readout layers.
            dropout (float): Dropout rate for regularization.
            lr (float): Learning rate for the optimizer.
        """
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
                                      atom_pool_hidden_dim=atom_pool_hidden_dim,
                                      atom_embedding_dim=atom_embedding_dim,
                                      res_embedding_dim=res_embedding_dim,
                                      res_edge_embedding_dim=res_edge_embedding_dim,
                                      num_atom_pool_layers=num_atom_pool_layers,
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
        del atom_cross_dists, atom_edge_index, atom_mask, atom_coords
        torch.cuda.empty_cache()
        res_embedding, res_edge_embedding, edge_index = self.res_feature(coords, mask, atom_embedding)
        del atom_embedding
        torch.cuda.empty_cache()
        for layer in self.res_mpnn_layers:
            res_embedding, res_edge_embedding = layer(res_embedding, res_edge_embedding, edge_index, mask)
        del res_edge_embedding, edge_index
        torch.cuda.empty_cache()
        logits = self.readout(res_embedding, mask)

        return logits

    def training_step(self, batch):
        sequences, coords, mask, _ = batch
        sequences.to(self.device)
        coords = coords.to(self.device)
        mask = mask.to(self.device)
        logits = self(coords, mask)
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
        """
        Perform a test step, computing loss and sequence recovery rate.

        Args:
            batch: A batch of test data.

        Returns:
            dict: Test metrics including loss and recovery rate.
        """
        sequences, coords, mask, _ = batch
        sequences = sequences.to(self.device)
        coords = coords.to(self.device)
        mask = mask.to(self.device)

        # Forward pass
        logits = self(coords, mask)

        # Compute loss
        loss = self.loss_fn(F.softmax(logits, dim=-1).view(-1), sequences.view(-1))
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)

        # Compute sequence recovery rate
        probs = F.softmax(logits, dim=-1)
        correct = (probs.argmax(dim=-1) == sequences.argmax(dim=-1)) * mask
        recovery_rate = correct.sum().item() / mask.sum().item()
        self.log('test_recovery_rate', recovery_rate, prog_bar=True, sync_dist=True)

        return {'test_loss': loss, 'test_recovery_rate': recovery_rate}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
        return [optimizer], [scheduler]

    def predict(self, batch, batch_id, output_dir, filename):
        """
        Perform inference and save results to a CSV file.

        Args:
            batch: A batch of input data.
            batch_id: The ID of the current batch.
            output_dir: The directory to save the output file.
            filename: The name of the output CSV file.
        """# Reverse the VOCAB dictionary

        _, coords, mask, pdb_id = batch
        coords = coords.to(self.device)
        mask = mask.to(self.device)

        # Perform inference
        logits = self(coords, mask)
        predictions = torch.argmax(logits, dim=-1)  # Shape: (batch_size, max_len)

        # Decode predictions to RNA sequences
        rna_sequences = []
        for i in range(predictions.size(0)):
            seq = "".join([REVERSE_VOCAB[idx] for idx in predictions[i][mask[i] == 1].tolist()])
            rna_sequences.append(seq)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Write results to CSV
        write_header = batch_id == 0  # Write header only for the first batch
        with open(output_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["pdb_id", "seq"])  # Write header
            for pdb, seq in zip(pdb_id, rna_sequences):
                writer.writerow([pdb, seq])

