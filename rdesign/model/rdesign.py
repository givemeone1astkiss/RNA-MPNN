import torch
from typing import Any
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn.functional as F
from typing_extensions import final
import os
from ..config.glob import NUM_RES_TYPES, DEFAULT_SEED, OUTPUT_PATH
from ..utils.data import separate
import pytorch_lightning as pl
import xgboost as xgb
import pickle
import sklearn
from .mpnn import MPNNLayer
from .feature import RNAFeatures
from .functional import Readout, RNABert

@final
class RNAModel(pl.LightningModule):
    def __init__(self,
                hidden_dim: int = 128,
                vocab_size: int = 4,
                k_neighbors: int = 25,
                dropout: float = 0.1,
                node_feat_types=None,
                edge_feat_types=None,
                num_message_layers: int = 3,
                num_dense_layers: int = 3,
                dim_dense_layers: int = 256,
                num_mpnn_layers: int = 9,
                readout_hidden_dim: int = 256,
                num_readout_layers: int = 0,
                lr: float = 0.002,
                n_estimators: int = 100,
                xgb_max_depth: int = 6,
                xgb_learning_rate: float = 0.1,
                xgb_subsample: float = 0.8,
                xgb_colsample_bytree: float = 0.8,):
        super().__init__()

        if edge_feat_types is None:
            edge_feat_types = ['orientation', 'distance', 'direction']
        if node_feat_types is None:
            node_feat_types = ['angle', 'distance', 'direction']

        self.name = 'RDesign-X'
        self.version = 0
        self.save_hyperparameters()
        self.node_features = self.edge_features = hidden_dim
        self.hidden_dim = hidden_dim
        self.vocab = vocab_size

        self.features = RNAFeatures(
            hidden_dim, hidden_dim,
            top_k=k_neighbors,
            dropout=dropout,
            node_feat_types=node_feat_types,
            edge_feat_types=edge_feat_types,
        )

        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(self.hidden_dim, self.hidden_dim*2, num_message_layers, num_dense_layers, dim_dense_layers, dropout=dropout)
            for _ in range(num_mpnn_layers)])

        self.readout = Readout(hidden_dim, readout_hidden_dim, num_readout_layers, dropout=dropout)

        self.xgb_readout = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=NUM_RES_TYPES,
            n_estimators=n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            random_state=DEFAULT_SEED
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.val_step_outputs = {'val_loss':[], 'correct':[], 'len':[], 'recovery_rates':[]}
        self.test_step_outputs = {'test_loss':[], 'correct':[], 'len':[], 'recovery_rates':[]}

    def forward(self, X, S, mask, is_predict=False):
        _, S, h_V, h_E, E_idx, _ = self.features(X, S, mask)
        for layer in self.mpnn_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = layer(h_V, h_EV, E_idx)

        return h_V, S

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
        return [optimizer], [scheduler]

    def training_step(self, batch) -> STEP_OUTPUT:
        X, S, mask, lengths, _ = batch
        X = X.to(self.device)
        S = S.to(self.device)
        mask = mask.to(self.device)
        h_V, S = self(X, S, mask)
        logits = self.readout(h_V)
        loss = self.loss_fn(logits, S)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch) -> STEP_OUTPUT:
        X, S, mask, lengths, _ = batch
        X = X.to(self.device)
        S = S.to(self.device)
        mask = mask.to(self.device)
        h_V, S = self(X, S, mask)
        logits = self.readout(h_V)
        loss = self.loss_fn(logits, S)
        probs = F.softmax(logits, dim=-1)
        samples = probs.argmax(dim=-1)
        correct = (samples == S).to(dtype=torch.float32)
        sep_correct = separate(correct, lengths).to(device=self.device)
        recovery_rates = (sep_correct.sum(dim=-1) / torch.tensor(lengths).to(device=self.device)).tolist()
        self.val_step_outputs['val_loss'].append(loss * correct.shape[0])
        self.val_step_outputs['correct'].append(correct.sum(dim=-1).item())
        self.val_step_outputs['len'].append(correct.shape[0])
        self.val_step_outputs['recovery_rates'] += recovery_rates
        return {'validation loss': loss, 'recovery_rates': recovery_rates}

    def test_step(self, batch) -> STEP_OUTPUT:
        X, S, mask, lengths, _ = batch
        X = X.to(self.device)
        S = S.to(self.device)
        mask = mask.to(self.device)
        h_V, S = self(X, S, mask)
        logits = self.readout(h_V)
        loss = self.loss_fn(logits, S)
        probs = F.softmax(logits, dim=-1)
        samples = probs.argmax(dim=-1)
        correct = (samples == S).to(dtype=torch.float32)
        sep_correct = separate(correct, lengths).to(device=self.device)
        recovery_rates = (sep_correct.sum(dim=-1) / torch.tensor(lengths).to(device=self.device)).tolist()
        self.val_step_outputs['val_loss'].append(loss * correct.shape[0])
        self.val_step_outputs['correct'].append(correct.sum(dim=-1).item())
        self.val_step_outputs['len'].append(correct.shape[0])
        self.val_step_outputs['recovery_rates'] += recovery_rates
        return {'test loss': loss, 'recovery_rates': recovery_rates}

    def predict(self, batch, batch_id, output_dir, filename):
        self.eval()
        X, S, mask, lengths, pdb_ids = batch
        X = X.to(self.device)
        S = S.to(self.device)
        mask = mask.to(self.device)

        h_V, _ = self(X, S, mask)
        try:
            samples = self.xgb_readout.predict(h_V.to(device=torch.device(torch.device('cpu'))).detach().numpy())
        except sklearn.exceptions.NotFittedError:
            samples = self.readout(h_V).argmax(dim=-1)
        start_idx = 0
        sequences = []
        for length, pdb_id in zip(lengths, pdb_ids):
            end_idx = start_idx + length.item()
            sample = samples[start_idx:end_idx]
            seq = ''.join(['AUCG'[i] for i in sample.tolist()])
            sequences.append((pdb_id, seq))
            start_idx = end_idx

        # Write to CSV
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'a') as f:
            if batch_id == 0:
                f.write("pdb_id,seq\n")
            for pdb_id, seq in sequences:
                f.write(f"{pdb_id},{seq}\n")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Save model-specific attributes to the checkpoint.

        Args:
            checkpoint (dict): The checkpoint dictionary.
        """
        checkpoint['name'] = self.name
        checkpoint['version'] = self.version

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Load the model from a checkpoint.

        Args:
            checkpoint (dict): The checkpoint dictionary containing the model state.
        """
        super().on_load_checkpoint(checkpoint)
        self.name = checkpoint.get('name', 'RNAMPNN-X')
        self.version = checkpoint.get('version', 0)
        self.xgb_readout = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=NUM_RES_TYPES,
            n_estimators=self.hparams.n_estimators,
            max_depth=self.hparams.xgb_max_depth,
            learning_rate=self.hparams.xgb_learning_rate,
            subsample=self.hparams.xgb_subsample,
            colsample_bytree=self.hparams.xgb_colsample_bytree,
            random_state=DEFAULT_SEED
        )
        try:
            with open(f"{OUTPUT_PATH}/checkpoints/{self.name}/XGB-V{self.version}.pkl", 'rb') as f:
                self.xgb_readout = pickle.load(f)
        except FileNotFoundError:
            print("=========  XGBoost model not found. Please train the model first.  =========")
        print("=========  Model loaded successfully from checkpoint.  =========")