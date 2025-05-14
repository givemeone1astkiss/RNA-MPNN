from rnampnn.model.mpnn import ResMPNN, AtomMPNN
from rnampnn.model.feature import ResFeature, AtomFeature, to_atom_format
from rnampnn.model.functional import Readout, BertReadout, BertEmbedding
from rnampnn.model.rnampnn import RNAMPNN
import torch
from rnampnn.utils.data import RNADataModule
from tqdm import tqdm
import torch.nn.functional as F


def test_res_feature():
    coords = torch.tensor([
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0], [2.0, 1.0, 1.732],[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ]
        ])
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0]
    ])

    res_feature = ResFeature(num_neighbours=5, num_cross_dihedral_atoms=6, num_cross_dist_atoms=6,
                             num_cross_angle_atoms=6, num_inside_dist_atoms=6, num_inside_angle_atoms=6, num_inside_dihedral_atoms=6)

    res_embedding, res_edge_embedding, edge_index = res_feature(coords, mask)
    print('\n','='*79)
    print(res_embedding.shape, res_edge_embedding.shape, edge_index)
    print('='*79,'\n')

def test_atom_feature():
    coords = torch.tensor([
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
             [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]
        ]
    ])
    mask = torch.tensor([
        [1, 0]
    ])
    atom_coords, atom_mask = to_atom_format(coords, mask)
    atom_feature = AtomFeature(num_atom_neighbours=7)
    atom_embedding, atom_cross_dists, atom_edge_index = atom_feature(atom_coords, atom_mask)
    print('\n','='*79)
    print(atom_embedding.shape, atom_cross_dists.shape, atom_edge_index.shape)
    print('='*79,'\n')

def test_res_mpnn():
    coords = torch.tensor([
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
             [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [2.0, 1.0, 1.732], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]
        ]
    ])
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0]
    ])

    res_feature = ResFeature(num_neighbours=3, num_cross_dihedral_atoms=7, num_cross_dist_atoms=7,
                             num_cross_angle_atoms=7, num_inside_dist_atoms=6, num_inside_angle_atoms=6,
                             num_inside_dihedral_atoms=6, res_embedding_dim=32, res_edge_embedding_dim=32)

    res_embedding, res_edge_embedding, edge_index = res_feature(coords, mask)
    res_mpnn = ResMPNN(res_embedding_dim=32, res_edge_embedding_dim=32, depth_res_mpnn=2, num_edge_layers=2)
    message = res_mpnn.message(res_embedding, res_edge_embedding, edge_index, mask)
    res_embedding, res_edge_embedding = res_mpnn(res_embedding, res_edge_embedding, edge_index, mask)
    print('\n', '=' * 79)
    print(message.shape)
    print(res_embedding.shape, res_edge_embedding.shape)
    print('=' * 79, '\n')

def test_atom_mpnn():
    coords = torch.tensor([
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
             [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]
        ],
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
             [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]
        ]
    ])
    mask = torch.tensor([
        [1, 0], [1,0]
    ])
    atom_coords, atom_mask = to_atom_format(coords, mask)
    atom_feature = AtomFeature(num_atom_neighbours=5, atom_embedding_dim=32)
    atom_mpnn = AtomMPNN(atom_embedding_dim=32, depth_atom_mpnn=2)
    atom_embedding, atom_cross_dists, atom_edge_index = atom_feature(atom_coords, atom_mask)
    print(atom_embedding.shape, atom_cross_dists.shape, atom_edge_index.shape)
    message = atom_mpnn.message(atom_embedding, atom_cross_dists, atom_edge_index, atom_mask)
    atom_embedding = atom_mpnn(atom_embedding, atom_cross_dists, atom_edge_index, atom_mask)
    print('\n', '=' * 79)
    print(message.shape)
    print(atom_embedding.shape)
    print('=' * 79, '\n')

def test_readout():
    res_embedding = torch.cat((torch.rand(1, 30, 32), torch.zeros(1, 5, 32)), dim=1)
    mask = torch.cat((torch.ones(1, 30), torch.zeros(1, 5)), dim=1)
    readout = Readout(res_embedding_dim=32, num_layers=2, readout_hidden_dim=32)

    logits = readout(res_embedding, mask)

    print('\n', '=' * 79)
    print(logits.shape)
    print('=' * 79, '\n')

def test_bert_readout():
    res_embedding = torch.cat((torch.rand(1, 30, 32), torch.zeros(1, 5, 32)), dim=1)
    mask = torch.cat((torch.ones(1, 30), torch.zeros(1, 5)), dim=1)
    readout = BertReadout(res_embedding_dim=32, padding_len=40, num_attn_layers=2, num_heads=4, num_ffn_layers=2, ffn_dim=32)

    logits = readout(res_embedding, mask)

    print('\n', '=' * 79)
    print(logits.shape)
    print('=' * 79, '\n')

def test_bert_embedding():
    raw_feature = torch.cat((torch.rand(1, 30, 8), torch.zeros(1, 5, 8)), dim=1)
    mask = torch.cat((torch.ones(1, 30), torch.zeros(1, 5)), dim=1)
    embedding = BertEmbedding(raw_dim=8, res_embedding_dim=32, padding_len=40, num_attn_layers=2, num_heads=4, num_ffn_layers=2, ffn_dim=32)
    res_embedding = embedding(raw_feature, mask)
    print('\n', '=' * 79)
    print(res_embedding)
    print('=' * 79, '\n')

def test_data_loader():
    rna_datamodule = RNADataModule()
    rna_datamodule.setup()
    train_loader = rna_datamodule.train_dataloader()
    for batch in train_loader:
        print(batch)
        break

def test_module():
    rna_datamodule = RNADataModule(batch_size=4)
    rna_datamodule.setup()
    train_loader = rna_datamodule.train_dataloader()
    rna_module = RNAMPNN()
    for batch in train_loader:
        loss = rna_module.validation_step(batch)
        print(loss)
        break

def test_nan():
    rna_datamodule = RNADataModule(batch_size=4)
    rna_datamodule.setup()
    train_loader = rna_datamodule.train_dataloader()
    rna_module = RNAMPNN()
    for batch in train_loader:
        loss = rna_module.validation_step(batch)
        print(loss)
        break

def test_grad():
    model = RNAMPNN().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    data = RNADataModule(split_ratio=0.95, batch_size=2, min_len=2)
    data.setup()
    for batch in tqdm(data.train_dataloader(), desc="Training", unit="batch"):
        batch = batch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        loss = model.training_step(batch)
        model.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients

        # Check for zero gradients
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Warning: Gradient for parameter '{name}' is None.")
            elif torch.all(param.grad == 0):
                print(f"Warning: Gradient for parameter '{name}' is all zeros.")

def test_quaternions():
    coords = torch.tensor([
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
             [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [2.0, 1.0, 1.732], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]
        ]
    ])
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0]
    ])
    res_feature = ResFeature(num_neighbours=5, num_cross_dihedral_atoms=6, num_cross_dist_atoms=6, num_cross_angle_atoms=6, num_inside_dist_atoms=6, num_inside_angle_atoms=6, num_inside_dihedral_atoms=6)
    edge_index = res_feature._get_res_graph(coords, mask)
    quaternions = res_feature._quaternions(coords, mask, edge_index)
    print(quaternions.shape)

def test_recovery():
    coords = torch.tensor([
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
             [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [2.0, 1.0, 1.732], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]
        ],
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
             [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 1.732], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [2.0, 1.0, 1.732], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]
        ]
    ])
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0],
    ])
    sequences = torch.tensor([
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
    ])
    model = RNAMPNN()
    logits = model(coords, mask)
    probs = F.softmax(logits, dim=-1)
    valid_probs = probs[mask.bool()]
    valid_sequences = sequences[mask.bool()]
    correct = (valid_sequences.argmax(dim=-1) == valid_probs.argmax(dim=-1)).to(dtype=float)
    print(correct)
    print(correct.sum(dim=-1).item()/correct.shape[0])



if __name__ == "__main__":
    pass