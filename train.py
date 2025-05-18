from rnampnn.utils.data import RNADataModule
from rnampnn.utils.train import get_trainer
from rnampnn.model.rnampnn import RNAMPNN
from rnampnn.utils.seed import seeding
import xgboost as xgb
from rnampnn.config.glob import DEFAULT_SEED, NUM_RES_TYPES, NUM_MAIN_SEQ_ATOMS, DEFAULT_HIDDEN_DIM

seeding()

model = RNAMPNN(num_res_neighbours = 4,
                 num_inside_dist_atoms = NUM_MAIN_SEQ_ATOMS,
                 num_inside_angle_atoms = NUM_MAIN_SEQ_ATOMS - 1,
                 num_inside_dihedral_atoms = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dist_atoms = NUM_MAIN_SEQ_ATOMS,
                 num_cross_angle_atoms = NUM_MAIN_SEQ_ATOMS - 1,
                 num_cross_dihedral_atoms = NUM_MAIN_SEQ_ATOMS - 1,
                 res_embedding_dim = DEFAULT_HIDDEN_DIM,
                 num_embedding_attn_layers = 2,
                 num_embedding_heads = 6,
                 embedding_ffn_dim = 256,
                 num_embedding_ffn_layers = 1,
                 res_edge_embedding_dim = DEFAULT_HIDDEN_DIM,
                 depth_res_edge_feature = 2,
                 num_res_mpnn_layers = 8,
                 depth_res_mpnn = 2,
                 num_mpnn_edge_layers = 2,
                 padding_len = 4500,
                 num_post_fusion_attn_layers = 2,
                 num_post_fusion_heads = 8,
                 post_fusion_ffn_dim = 256,
                 num_post_fusion_ffn_layers = 1,
                 num_raw_ffn_layers = 2,
                 num_raw_ffn_dim=256,
                 raw_embedding_dim=DEFAULT_HIDDEN_DIM,
                 readout_hidden_dim = 256,
                 num_readout_layers = 1,
                 dropout = 0.2,
                 lr = 2.5e-3,
                 weight_decay = 0.00001,
                 n_estimators=150,
                 xgb_max_depth=10,
                 xgb_learning_rate=0.1,
                 xgb_subsample=0.8,
                 xgb_colsample_bytree=0.8,)

# model = RNAMPNN.load_from_checkpoint('out/checkpoints/RNAMPNN-X/Final-V0.ckpt')
# model.xgb_readout = xgb.XGBClassifier(objective='multi:softmax',
#                                     num_class=NUM_RES_TYPES,
#                                     n_estimators=180,
#                                     xgb_max_depth=8,
#                                     xgb_learning_rate=0.1,
#                                     xgb_subsample=0.9,
#                                     xgb_colsample_bytree=0.8,
#                                     random_state= DEFAULT_SEED,
#                                     n_jobs=8,
#                                     )
data = RNADataModule(split_ratio=0.95, batch_size=3)
trainer = get_trainer(name='RNAMPNN-X', version=1, max_epochs=180)
trainer.fit(model, data)


# from rdesign.model.rdesign import RNAModel
# from rdesign.utils.data import RNADataModule
# from rdesign.utils.train import get_trainer
# from rdesign.config.seeds import seeding
#
# seeding()
# model = RNAModel()
# data = RNADataModule.from_defaults(batch_size=16, split_ratio=[0.9, 0.1, 0.0])
# trainer = get_trainer(name='RDesign-X', version=1, max_epochs=80)
# trainer.fit(model, data)
