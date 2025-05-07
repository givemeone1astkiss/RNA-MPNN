from rnampnn.model.rnampnn import ResFeature, AtomFeature, to_atom_format, ResMPNN, AtomMPNN, Readout, RNAMPNN
import torch
from rnampnn.utils.data import RNADataModule


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

    atom_embedding = torch.cat((torch.rand(1, 35, 32), torch.zeros(1, 7, 32)), dim=1)

    res_feature = ResFeature(num_neighbours=7, num_cross_dihedral_atoms=7, num_cross_dist_atoms=7, atom_embedding_dim=32,
                             num_cross_angle_atoms=7, num_inside_dist_atoms=6, num_inside_angle_atoms=6, num_inside_dihedral_atoms=6)

    res_embedding, res_edge_embedding, edge_index = res_feature(coords, mask, atom_embedding)
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

    atom_embedding = torch.cat((torch.rand(1, 35, 32), torch.zeros(1, 7, 32)), dim=1)

    res_feature = ResFeature(num_neighbours=3, num_cross_dihedral_atoms=7, num_cross_dist_atoms=7,
                             atom_embedding_dim=32,
                             num_cross_angle_atoms=7, num_inside_dist_atoms=6, num_inside_angle_atoms=6,
                             num_inside_dihedral_atoms=6, res_embedding_dim=32, res_edge_embedding_dim=32)

    res_embedding, res_edge_embedding, edge_index = res_feature(coords, mask, atom_embedding)
    res_mpnn = ResMPNN(res_embedding_dim=32, res_edge_embedding_dim=32, depth_res_mpnn=2, num_edge_layers=2)
    message = res_mpnn.message(res_embedding, res_edge_embedding, edge_index, mask)
    res_embedding, res_edge_embedding, edge_index, mask = res_mpnn(res_embedding, res_edge_embedding, edge_index, mask)
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
    atom_embedding, atom_cross_dists, atom_edge_index, atom_mask = atom_mpnn(atom_embedding, atom_cross_dists, atom_edge_index, atom_mask)
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
    
    
if __name__ == '__main__':
    # test_res_feature()
    # test_atom_feature()
    # test_res_mpnn()
    # test_atom_mpnn()
    # test_readout()
    # test_data_loader()
    test_module()