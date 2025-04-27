import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch_scatter import scatter_sum
import numpy as np
import torch.nn.functional as F
from typing_extensions import final
import os
from ..config.glob import FEAT_DIMS
from ..utils.data import normalize
import pytorch_lightning as pl

class MPNNLayer(nn.Module):
    """
    Implements a Message Passing Neural Network (MPNN) layer for graph-based learning.

    Attributes:
        num_hidden (int): Number of hidden units in the layer.
        num_in (int): Number of input features for edges.
        scale (int): Scaling factor for message aggregation.
        dropout (nn.Dropout): Dropout layer for regularization.
        norm1 (nn.LayerNorm): Layer normalization for the first residual connection.
        norm2 (nn.LayerNorm): Layer normalization for the second residual connection.
        W1 (nn.Linear): Linear transformation for edge features.
        W2 (nn.Linear): Linear transformation for intermediate features.
        W3 (nn.Linear): Linear transformation for message features.
        dense (nn.Sequential): Feedforward network for node updates.
    """
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        """
        Initializes the MPNN layer.

        Args:
            num_hidden (int): Number of hidden units in the layer.
            num_in (int): Number of input features for edges.
            dropout (float): Dropout rate for regularization.
            scale (int): Scaling factor for message aggregation.
        """
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.GELU(),
            nn.Linear(num_hidden * 4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx)->torch.Tensor:
        """
        Forward pass for the MPNN layer.

        Args:
            h_V (torch.Tensor): Node features of shape (num_nodes, num_hidden).
            h_E (torch.Tensor): Edge features of shape (num_edges, num_in).
            edge_idx (torch.Tensor): Edge indices of shape (2, num_edges).

        Returns:
            torch.Tensor: Updated node features of shape (num_nodes, num_hidden).
        """
        src_idx, _ = edge_idx[0], edge_idx[1]
        h_message = self.W3(F.gelu(self.W2(F.gelu(self.W1(h_E)))))
        dh = scatter_sum(h_message, src_idx, dim=0) / self.scale
        h_V = self.norm1(h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))
        return h_V

class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class RNAFeatures(nn.Module):
    def __init__(self, edge_features, node_features, node_feat_types=[], edge_feat_types=[], num_rbf=16, top_k=30,
                 augment_eps=0., dropout=0.1):
        super().__init__()
        """Extract RNA Features"""
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.dropout = nn.Dropout(dropout)
        self.node_feat_types = node_feat_types
        self.edge_feat_types = edge_feat_types

        node_in = sum([FEAT_DIMS['node'][feat] for feat in node_feat_types])
        edge_in = sum([FEAT_DIMS['edge'][feat] for feat in edge_feat_types])
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    @staticmethod
    def _gather_nodes(nodes, neighbor_idx):
        neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
        return neighbor_features

    @staticmethod
    def _gather_edges(edges, neighbor_idx):
        neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
        return torch.gather(edges, 2, neighbors)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = (1. - mask_2D) * 10000 + mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max + 1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    def _get_rbf(self, A, B, E_idx=None, num_rbf=16):
        if E_idx is not None:
            D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
            D_A_B_neighbors = self._gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
            RBF_A_B = self._rbf(D_A_B_neighbors)
        else:
            D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, :, None, :]) ** 2, -1) + 1e-6)
            RBF_A_B = self._rbf(D_A_B)
        return RBF_A_B

    def _quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        V = X.clone()
        X = X[:, :, :6, :].reshape(X.shape[0], 6 * X.shape[1], 3)
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = normalize(dX, dim=-1)
        u_0, u_1 = U[:, :-2, :], U[:, 1:-1, :]
        n_0 = normalize(torch.linalg.cross(u_0, u_1), dim=-1)
        b_1 = normalize(u_0 - u_1, dim=-1)

        # select C3
        n_0 = n_0[:, 4::6, :]
        b_1 = b_1[:, 4::6, :]
        X = X[:, 4::6, :]

        Q = torch.stack((b_1, n_0, torch.linalg.cross(b_1, n_0)), 2)
        Q = Q.view(list(Q.shape[:2]) + [9])
        Q = F.pad(Q, (0, 0, 0, 1), 'constant', 0)  # [16, 464, 9]

        Q_neighbors = self._gather_nodes(Q, E_idx)  # [16, 464, 30, 9]
        P_neighbors = self._gather_nodes(V[:, :, 0, :], E_idx)  # [16, 464, 30, 3]
        O5_neighbors = self._gather_nodes(V[:, :, 1, :], E_idx)
        C5_neighbors = self._gather_nodes(V[:, :, 2, :], E_idx)
        C4_neighbors = self._gather_nodes(V[:, :, 3, :], E_idx)
        O3_neighbors = self._gather_nodes(V[:, :, 5, :], E_idx)

        Q = Q.view(list(Q.shape[:2]) + [3, 3]).unsqueeze(2)  # [16, 464, 1, 3, 3]
        Q_neighbors = Q_neighbors.view(list(Q_neighbors.shape[:3]) + [3, 3])  # [16, 464, 30, 3, 3]

        dX = torch.stack([P_neighbors, O5_neighbors, C5_neighbors, C4_neighbors, O3_neighbors], dim=3) - X[:, :, None,
                                                                                                         None,
                                                                                                         :]  # [16, 464, 30, 3]
        dU = torch.matmul(Q[:, :, :, None, :, :], dX[..., None]).squeeze(-1)  # [16, 464, 30, 3] 邻居的相对坐标
        B, N, K = dU.shape[:3]
        E_direct = normalize(dU, dim=-1)
        E_direct = E_direct.reshape(B, N, K, -1)
        R = torch.matmul(Q.transpose(-1, -2), Q_neighbors)
        E_orient = self._quaternions(R)

        dX_inner = V[:, :, [0, 2, 3], :] - X.unsqueeze(-2)
        dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
        dU_inner = normalize(dU_inner, dim=-1)
        V_direct = dU_inner.reshape(B, N, -1)
        return V_direct, E_direct, E_orient

    def _dihedrals(self, X, eps=1e-7):
        # P, O5', C5', C4', C3', O3'
        X = X[:, :, :6, :].reshape(X.shape[0], 6 * X.shape[1], 3)

        dX = X[:, 5:, :] - X[:, :-5, :]  # O3'-P, P-O5', O5'-C5', C5'-C4', ...
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]  # O3'-P, P-O5', ...
        u_1 = U[:, 1:-1, :]  # P-O5', O5'-C5', ...
        u_0 = U[:, 2:, :]  # O5'-C5', C5'-C4', ...
        # Backbone normals
        n_2 = F.normalize(torch.linalg.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.linalg.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        D = F.pad(D, (3, 4), 'constant', 0)
        D = D.view((D.size(0), D.size(1) // 6, 6))
        return torch.cat((torch.cos(D), torch.sin(D)), 2)  # return D_features

    def forward(self, X, S, mask):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        B, N, _, _ = X.shape
        # P, O5', C5', C4', C3', O3'
        atom_P = X[:, :, 0, :]
        atom_O5_ = X[:, :, 1, :]
        atom_C5_ = X[:, :, 2, :]
        atom_C4_ = X[:, :, 3, :]
        atom_C3_ = X[:, :, 4, :]
        atom_O3_ = X[:, :, 5, :]

        X_backbone = atom_P
        D_neighbors, E_idx = self._dist(X_backbone, mask)

        mask_bool = (mask == 1)
        mask_attend = self._gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x: torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1, x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        # node features
        h_V = []
        # angle
        V_angle = node_mask_select(self._dihedrals(X))
        # distance
        node_list = ['O5_-P', 'C5_-P', 'C4_-P', 'C3_-P', 'O3_-P']
        V_dist = []

        for pair in node_list:
            atom1, atom2 = pair.split('-')
            V_dist.append(node_mask_select(
                self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        V_dist = torch.cat(tuple(V_dist), dim=-1).squeeze()
        # direction
        V_direct, E_direct, E_orient = self._orientations_coarse(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct, E_orient = list(map(lambda x: edge_mask_select(x), [E_direct, E_orient]))

        # edge features
        h_E = []
        # dist
        edge_list = ['P-P', 'O5_-P', 'C5_-P', 'C4_-P', 'C3_-P', 'O3_-P']
        E_dist = []
        for pair in edge_list:
            atom1, atom2 = pair.split('-')
            E_dist.append(
                edge_mask_select(self._get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)))
        E_dist = torch.cat(tuple(E_dist), dim=-1)

        if 'angle' in self.node_feat_types:
            h_V.append(V_angle)
        if 'distance' in self.node_feat_types:
            h_V.append(V_dist)
        if 'direction' in self.node_feat_types:
            h_V.append(V_direct)

        if 'orientation' in self.edge_feat_types:
            h_E.append(E_orient)
        if 'distance' in self.edge_feat_types:
            h_E.append(E_dist)
        if 'direction' in self.edge_feat_types:
            h_E.append(E_direct)

        # Embed the nodes
        h_V = self.norm_nodes(self.node_embedding(torch.cat(h_V, dim=-1)))
        h_E = self.norm_edges(self.edge_embedding(torch.cat(h_E, dim=-1)))

        # prepare the variables to return
        S = torch.masked_select(S, mask_bool)
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B, 1, 1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1, -1)
        dst = shift.view(B, 1, 1) + torch.arange(0, N, device=src.device).view(1, -1, 1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1, -1)
        E_idx = torch.cat((dst, src), dim=0).long()

        sparse_idx = mask.nonzero()
        X = X[sparse_idx[:, 0], sparse_idx[:, 1], :, :]
        batch_id = sparse_idx[:, 0]
        return X, S, h_V, h_E, E_idx, batch_id

@final
class RNAModel(pl.LightningModule):
    """
    RNA Model for sequence generation using MPNN.
    Attributes:
        hidden (int): Number of hidden units.
        vocab_size (int): Size of the vocabulary.
        k_neighbors (int): Number of nearest neighbors.
        dropout (float): Dropout rate.
        node_feat_types (list): Types of node features to include.
        edge_feat_types (list): Types of edge features to include.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        lr (float): Learning rate.
    """
    def __init__(self,
                hidden_dim: int = 128,
                vocab_size: int = 4,
                k_neighbors: int = 40,
                dropout: float = 0.1,
                node_feat_types=None,
                edge_feat_types=None,
                num_mpnn_layers: int = 9,
                lr: float = 0.001
                ):
        """
        Initializes the RNAModel.
        Args:
            hidden_dim (int): Number of hidden units.
            vocab_size (int): Size of the vocabulary.
            k_neighbors (int): Number of nearest neighbors.
            dropout (float): Dropout rate.
            node_feat_types (list): Types of node features to include.
            edge_feat_types (list): Types of edge features to include.
            num_mpnn_layers (int): Number of mpnn layers.
            lr (float): Learning rate.
        """
        super().__init__()

        if edge_feat_types is None:
            edge_feat_types = ['orientation', 'distance', 'direction']
        if node_feat_types is None:
            node_feat_types = ['angle', 'distance', 'direction']
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
            MPNNLayer(self.hidden_dim, self.hidden_dim*2, dropout=dropout)
            for _ in range(num_mpnn_layers)])

        self.readout = nn.Linear(self.hidden_dim, vocab_size, bias=True)
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, X, S, mask):
        X, S, h_V, h_E, E_idx, batch_id = self.features(X, S, mask)
        for layer in self.mpnn_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = layer(h_V, h_EV, E_idx)
        logits = self.readout(h_V)
        return logits, S

    def sample(self, X, S, mask=None):
        X, gt_S, h_V, h_E, E_idx, _ = self.features(X, S, mask)
        for layer in self.mpnn_layers:
            h_EV = torch.cat([h_E, h_V[E_idx[0]], h_V[E_idx[1]]], dim=-1)
            h_V = layer(h_V, h_EV, E_idx)
        logits = self.readout(h_V)
        return logits, gt_S

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Ruturn:
            torch.optim.Adam: Adam optimizer with the specified learning rate.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch) -> STEP_OUTPUT:
        X, S, mask, _, _ = batch
        X = X.to(self.device)
        S = S.to(self.device)
        mask = mask.to(self.device)
        logits, S = self(X, S, mask)
        loss = self.loss_fn(logits, S)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch) -> STEP_OUTPUT:
        X, S, mask, lengths, _ = batch
        X = X.to(self.device)
        S = S.to(self.device)
        mask = mask.to(self.device)
        logits, S = self(X, S, mask)
        loss = self.loss_fn(logits, S)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        probs = F.softmax(logits, dim=-1)
        samples = probs.argmax(dim=-1)

        recovery_list = []
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length.item()
            sample = samples[start_idx:end_idx]
            gt_S = S[start_idx:end_idx]
            recovery = (sample == gt_S).sum().item() / len(sample)
            recovery_list.append(recovery)
            start_idx = end_idx

        valid_recovery = np.mean(recovery_list)
        self.log('valid_recovery_rate', valid_recovery, prog_bar=True, sync_dist=True)
        return {'val_loss': loss, 'valid_recovery_rate': valid_recovery}

    def test_step(self, batch) -> STEP_OUTPUT:
        X, S, mask, lengths, _ = batch
        X = X.to(self.device)
        S = S.to(self.device)
        mask = mask.to(self.device)

        logits, S = self(X, S, mask)
        loss = self.loss_fn(logits, S)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)

        probs = F.softmax(logits, dim=-1)
        samples = probs.argmax(dim=-1)

        recovery_list = []
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length.item()
            sample = samples[start_idx:end_idx]
            gt_S = S[start_idx:end_idx]
            recovery = (sample == gt_S).sum().item() / len(sample)
            recovery_list.append(recovery)
            start_idx = end_idx

        valid_recovery = np.mean(recovery_list)
        self.log('test_recovery_rate', valid_recovery, prog_bar=True, sync_dist=True)
        return {'test_loss': loss, 'test_recovery_rate': valid_recovery}

    def predict(self, batch, batch_id, output_dir, filename):
        """
        Generate predictions for a batch of data and save them to a CSV file.

        Args:
            batch (tuple): A tuple containing the input data (X, S, mask, lengths, pdb_ids).
            batch_id (int): The ID of the current batch.
            output_dir (str): The directory to save the output CSV file.
            filename (str): The name of the output CSV file.
        """
        self.eval()
        X, S, mask, lengths, pdb_ids = batch
        X = X.to(self.device)
        S = S.to(self.device)
        mask = mask.to(self.device)

        logits, _ = self.sample(X, S, mask)
        probs = F.softmax(logits, dim=-1)
        samples = probs.argmax(dim=-1)
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
