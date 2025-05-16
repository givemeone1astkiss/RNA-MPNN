from rnampnn.utils.data import RNADataModule
from rnampnn.utils.train import get_trainer
from rnampnn.model.rnampnn import RNAMPNN
from rnampnn.utils.seed import seeding
from rnampnn.config.glob import DEFAULT_HIDDEN_DIM, NUM_MAIN_SEQ_ATOMS

seeding()
model = RNAMPNN(num_res_neighbours = 3,
                 num_inside_dist_atoms = NUM_MAIN_SEQ_ATOMS,
                 num_inside_angle_atoms = NUM_MAIN_SEQ_ATOMS - 1,
                 num_inside_dihedral_atoms = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dist_atoms = NUM_MAIN_SEQ_ATOMS,
                 num_cross_angle_atoms = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dihedral_atoms = NUM_MAIN_SEQ_ATOMS - 1,
                 res_embedding_dim = DEFAULT_HIDDEN_DIM,
                 num_embedding_attn_layers = 2,
                 num_embedding_heads = 8,
                 embedding_ffn_dim = 512,
                 num_embedding_ffn_layers = 2,
                 res_edge_embedding_dim = DEFAULT_HIDDEN_DIM,
                 depth_res_edge_feature = 2,
                 num_res_mpnn_layers = 10,
                 depth_res_mpnn = 2,
                 num_mpnn_edge_layers = 2,
                 padding_len = 4500,
                 num_readout_attn_layers = 2,
                 num_readout_heads = 8,
                 readout_ffn_dim = 512,
                 num_readout_ffn_layers = 3,
                 dropout = 0.4,
                 lr = 2e-3,
                 weight_decay = 0.001)

data = RNADataModule(split_ratio=0.95, batch_size=3)
trainer = get_trainer(name='RNAMPNN-AF', version=9, max_epochs=200)
trainer.fit(model, data)

# from rdesign.model.rdesign import RNAModel
# from rdesign.utils.data import RNADataModule
# from rdesign.utils.train import get_trainer
# from rdesign.config.seeds import seeding
# 
# seeding()
# model = RNAModel.load_from_checkpoint('out/checkpoints/RDesign/checkpoint-epoch=66-3.ckpt')
# data = RNADataModule.from_defaults(batch_size=16, split_ratio=[0.95, 0.05, 0.0])
# trainer = get_trainer(name='RDesign', version=3, max_epochs=200)
# trainer.fit(model, data)