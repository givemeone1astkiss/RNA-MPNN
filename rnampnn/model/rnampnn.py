import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import os
import csv
from ..config.glob import NUM_MAIN_SEQ_ATOMS, DEFAULT_HIDDEN_DIM, REVERSE_VOCAB, NUM_RES_TYPES, DEFAULT_SEED
from torch.nn import functional as F
from .mpnn import  ResMPNN
from .feature import  ResFeature
from .functional import BertReadout
import xgboost as xgb


class RNAMPNN(LightningModule):
    def __init__(self,
                 num_res_neighbours: int = 3,
                 num_inside_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_inside_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_inside_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dist_atoms: int = NUM_MAIN_SEQ_ATOMS,
                 num_cross_angle_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dihedral_atoms: int = NUM_MAIN_SEQ_ATOMS - 1,
                 res_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 num_embedding_attn_layers: int = 0,
                 num_embedding_heads: int = 8,
                 embedding_ffn_dim: int = 512,
                 num_embedding_ffn_layers: int = 3,
                 res_edge_embedding_dim: int = DEFAULT_HIDDEN_DIM,
                 depth_res_edge_feature: int = 2,
                 num_res_mpnn_layers: int = 10,
                 depth_res_mpnn: int = 2,
                 num_mpnn_edge_layers: int = 2,
                 padding_len: int = 4500,
                 num_readout_attn_layers: int = 2,
                 num_readout_heads: int = 8,
                 readout_ffn_dim: int = 512,
                 num_readout_ffn_layers: int = 3,
                 dropout: float = 0.4,
                 lr: float = 2e-3,
                 weight_decay: float = 0.0002,
                 n_estimators=100,
                 xgb_max_depth=6,
                 xgb_learning_rate=0.1,
                 xgb_subsample=0.8,
                 xgb_colsample_bytree=0.8,):
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
            n_estimators (int): Number of trees in the XGBoost model.
            xgb_max_depth (int): Maximum depth of the trees in the XGBoost model.
            xgb_learning_rate (float): Learning rate for the XGBoost model.
            xgb_subsample (float): Subsample ratio of the training instances for the XGBoost model.
            xgb_colsample_bytree (float): Subsample ratio of columns when constructing each tree for the XGBoost model.
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

        self.xgb_readout = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=NUM_RES_TYPES,
            n_estimators=n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            random_state= DEFAULT_SEED
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.val_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
        return [optimizer], [scheduler]

    def forward(self, coords: torch.Tensor, mask: torch.Tensor, is_predict: bool=False) -> torch.Tensor:
        """
        Forward pass for the RNAMPNN model.

        Args:
            coords (torch.Tensor): Atom coordinates of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
            mask (torch.Tensor): Mask indicating valid residues of shape (batch_size, max_len).
            is_predict (bool): Whether to clean cuda cache to save cuda memory.

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
        probs = F.softmax(logits, dim=-1)
        valid_probs = probs[mask.bool()]
        valid_sequences = sequences[mask.bool()]
        loss = self.loss_fn(valid_probs, valid_sequences) + self.loss_fn(valid_probs.reshape(valid_probs.shape[0], 2, 2).sum(dim=-1), valid_sequences.reshape(valid_sequences.shape[0], 2, 2).sum(dim=-1))
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
        probs = F.softmax(logits, dim=-1)
        valid_probs = probs[mask.bool()]
        valid_sequences = sequences[mask.bool()]
        loss = self.loss_fn(valid_probs, valid_sequences) + self.loss_fn(valid_probs.reshape(valid_probs.shape[0], 2, 2).sum(dim=-1), valid_sequences.reshape(valid_sequences.shape[0], 2, 2).sum(dim=-1))
        correct = (valid_probs.argmax(dim=-1) == valid_sequences.argmax(dim=-1)).to(dtype=torch.float32)
        self.val_step_outputs.append({'val_loss': loss, 'correct': correct.sum(dim=-1).item(), 'len': correct.shape[0]})
        return {'val_loss': loss, 'correct': correct.sum(dim=-1).item(), 'len': correct.shape[0]}


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
        probs = F.softmax(logits, dim=-1)
        valid_probs = probs[mask.bool()]
        valid_sequences = sequences[mask.bool()]
        loss = self.loss_fn(valid_probs, valid_sequences) + self.loss_fn(valid_probs.reshape(valid_probs.shape[0], 2, 2).sum(dim=-1), valid_sequences.reshape(valid_sequences.shape[0], 2, 2).sum(dim=-1))
        correct = (valid_probs.argmax(dim=-1) == valid_sequences.argmax(dim=-1)).to(dtype=torch.float32)
        self.test_step_output.append({'test_loss': loss, 'correct': correct.sum(dim=-1).item(), 'len': correct.shape[0]})
        return {'test_loss': loss, 'correct': correct.sum(dim=-1).item(), 'len': correct.shape[0]}

    def embedding(self, coords: torch.Tensor, mask: torch.Tensor, is_predict: bool=False) -> torch.Tensor:
        res_embedding, res_edge_embedding, edge_index = self.res_feature(coords, mask)
        if is_predict:
            del coords
            torch.cuda.empty_cache()
        for layer in self.res_mpnn_layers:
            res_embedding, _ = layer(res_embedding, res_edge_embedding, edge_index, mask)
        return  res_embedding

    def predict(self, batch, batch_id, output_dir, filename):
        """
        Perform inference and save results to a CSV file.

        Args:
            batch: A batch of input data.
            batch_id: The ID of the current batch.
            output_dir: The directory to save the output file.
            filename: The name of the output CSV file.
        """
        self.eval()
        _, coords, mask, pdb_id = batch
        coords = coords.to(self.device)
        mask = mask.to(self.device)

        # logits = self(coords, mask, is_predict=True)
        # predictions = torch.argmax(logits, dim=-1)  # Shape: (batch_size, max_len)
        embedding = self.embedding(coords, mask, is_predict=True)
        valid_embedding = embedding[mask.bool()]
        valid_predictions = self.xgb_readout.predict(valid_embedding.cpu())
        predictions = torch.zeros(coords.size(0), coords.size(1), dtype=torch.long).to(self.device)
        lengths = mask.sum(dim=-1)
        start = 0
        for i in range(coords.size(0)):
            end = int(lengths[i].item())
            predictions[i][start:end] = torch.tensor(valid_predictions[start:end], dtype=torch.long).to(self.device)
            start += end

        rna_sequences = []
        for i in range(predictions.size(0)):
            # Extract valid residues based on the mask
            valid_indices = mask[i] == 1
            seq = "".join([REVERSE_VOCAB[idx] for idx in predictions[i][valid_indices].tolist()])
            rna_sequences.append(seq)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        write_header = batch_id == 0  # Write header only for the first batch
        with open(output_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["pdb_id", "seq"])  # Write header
            for pdb, seq in zip(pdb_id, rna_sequences):
                writer.writerow([pdb, seq])


