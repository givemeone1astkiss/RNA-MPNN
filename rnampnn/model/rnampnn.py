import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import os
import csv
from ..config.glob import NUM_MAIN_SEQ_ATOMS, DEFAULT_HIDDEN_DIM, REVERSE_VOCAB
from torch.nn import functional as F
from .mpnn import  ResMPNN
from .feature import  ResFeature
from .functional import BertReadout, Readout


class RNAMPNN(LightningModule):
    def __init__(self,
                 num_res_neighbours: int = 30,
                 num_inside_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_inside_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_inside_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_cross_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 res_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 num_embedding_attn_layers: int = 2,
                 num_embedding_heads: int = 4,
                 embedding_ffn_dim: int = 512,
                 num_embedding_ffn_layers: int = 2,
                 res_edge_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 depth_res_feature: int = 2,
                 depth_res_edge_feature: int = 2,
                 num_res_mpnn_layers: int = 2,
                 depth_res_mpnn: int = 2,
                 num_mpnn_edge_layers: int = 2,
                 padding_len: int = 4500,
                 num_readout_attn_layers: int = 2,
                 num_readout_heads: int = 8,
                 readout_ffn_dim: int = 512,
                 num_readout_ffn_layers: int = 2,
                 dropout: float = 0.1,
                 lr: float = 2e-3,
                 weight_decay: float = 0.01):
        """
        Initialize the RNAMPNN model.

        Args:
            num_res_neighbours (int): Number of residue neighbours.
            num_inside_dist_atoms (int): Number of inside distance atoms.
            num_inside_angle_atoms (int): Number of inside angle atoms.
            num_inside_dihedral_atoms (int): Number of inside dihedral atoms.
            num_cross_dist_atoms (int): Number of cross distance atoms.
            num_cross_angle_atoms (int): Number of cross angle atoms.
            num_cross_dihedral_atoms (int): Number of cross dihedral atoms.
            res_embedding_dim (int): Dimension of residue embedding.
            num_embedding_attn_layers (int): Number of attention layers in the embedding module.
            num_embedding_heads (int): Number of attention heads in the embedding module.
            embedding_ffn_dim (int): Dimension of feedforward network in the embedding module.
            num_embedding_ffn_layers (int): Number of feedforward layers in the embedding module.
            res_edge_embedding_dim (int): Dimension of residue edge embedding.
            depth_res_feature (int): Depth of the residue feature extraction module.
            depth_res_edge_feature (int): Depth of the residue edge feature extraction module.
            num_res_mpnn_layers (int): Number of MPNN layers.
            depth_res_mpnn (int): Depth of the MPNN layers.
            num_mpnn_edge_layers (int): Number of edge layers in the MPNN.
            padding_len (int): Length of the padding for sequences.
            num_readout_attn_layers (int): Number of attention layers in the readout module.
            num_readout_heads (int): Number of attention heads in the readout module.
            readout_ffn_dim (int): Dimension of feedforward network in the readout module.
            num_readout_ffn_layers (int): Number of feedforward layers in the readout module.
            dropout (float): Dropout rate for regularization.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.res_feature = ResFeature(num_neighbours=num_res_neighbours,
                                      num_inside_dist_atoms=num_inside_dist_atoms,
                                      num_inside_angle_atoms=num_inside_angle_atoms,
                                      num_inside_dihedral_atoms=num_inside_dihedral_atoms,
                                      num_cross_dist_atoms=num_cross_dist_atoms,
                                      num_cross_angle_atoms=num_cross_angle_atoms,
                                      num_cross_dihedral_atoms=num_cross_dihedral_atoms,
                                      res_embedding_dim=res_embedding_dim,
                                      padding_len=padding_len,
                                      num_attn_layers=num_embedding_attn_layers,
                                      num_heads=num_embedding_heads,
                                      ffn_dim=embedding_ffn_dim,
                                      num_ffn_layers=num_embedding_ffn_layers,
                                      res_edge_embedding_dim=res_edge_embedding_dim,
                                      num_layers=depth_res_feature,
                                      num_edge_layers=depth_res_edge_feature,
                                      dropout=dropout)
        self.res_mpnn_layers = nn.ModuleList([ResMPNN(res_embedding_dim=res_embedding_dim, res_edge_embedding_dim=res_edge_embedding_dim, depth_res_mpnn=depth_res_mpnn, num_edge_layers=num_mpnn_edge_layers, dropout=dropout) for _ in range(num_res_mpnn_layers)])
        self.readout = BertReadout(padding_len=padding_len,
                                   res_embedding_dim=res_embedding_dim,
                                   num_attn_layers=num_readout_attn_layers,
                                   num_heads=num_readout_heads,
                                   ffn_dim=readout_ffn_dim,
                                   num_ffn_layers=num_readout_ffn_layers,
                                   dropout=dropout)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, coords: torch.Tensor, mask: torch.Tensor, is_predict: bool=False) -> torch.Tensor:
        """
        Forward pass for the RNAMPNN model.

        Args:
            coords (torch.Tensor): Atom coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask indicating valid residues of shape (batch_size, max_len).

        Returns:
            torch.Tensor: Predicted residue type logits of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS).
        """
        res_embedding, res_edge_embedding, edge_index = self.res_feature(coords, mask)
        if is_predict:
            del coords
            torch.cuda.empty_cache()
        for layer in self.res_mpnn_layers:
            res_embedding, res_edge_embedding = layer(res_embedding, res_edge_embedding, edge_index, mask)
        if is_predict:
            del res_edge_embedding, edge_index
            torch.cuda.empty_cache()
        logits = self.readout(res_embedding, mask)
        if is_predict:
            del res_embedding
            torch.cuda.empty_cache()
        return logits

    def training_step(self, batch):
        sequences, coords, mask, _ = batch
        sequences = sequences.to(self.device)
        coords = coords.to(self.device)
        mask = mask.to(self.device)

        logits = self(coords, mask)
        probs = F.softmax(logits)
        valid_probs = probs[mask.bool()]
        valid_sequences = sequences[mask.bool()]
        loss = self.loss_fn(valid_probs, valid_sequences)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, batch_size=2)

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

        logits = self(coords, mask)
        probs = F.softmax(logits)
        valid_probs = probs[mask.bool()]
        valid_sequences = sequences[mask.bool()]
        loss = self.loss_fn(valid_probs, valid_sequences)
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

        logits = self(coords, mask)
        probs = F.softmax(logits)
        valid_probs = probs[mask.bool()]
        valid_sequences = sequences[mask.bool()]
        loss = self.loss_fn(valid_probs, valid_sequences)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)

        # Compute sequence recovery rate
        probs = F.softmax(logits, dim=-1)
        correct = (probs.argmax(dim=-1) == sequences.argmax(dim=-1)) * mask
        recovery_rate = correct.sum().item() / mask.sum().item()
        self.log('test_recovery_rate', recovery_rate, prog_bar=True, sync_dist=True)

        return {'test_loss': loss, 'test_recovery_rate': recovery_rate}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
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
        """
        _, coords, mask, pdb_id = batch
        coords = coords.to(self.device)
        mask = mask.to(self.device)

        # Perform inference
        logits = self(coords, mask, is_predict=True)
        predictions = torch.argmax(logits, dim=-1)  # Shape: (batch_size, max_len)

        # Decode predictions to RNA sequences
        rna_sequences = []
        for i in range(predictions.size(0)):
            # Extract valid residues based on the mask
            valid_indices = mask[i] == 1
            seq = "".join([REVERSE_VOCAB[idx] for idx in predictions[i][valid_indices].tolist()])
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