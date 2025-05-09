# from rdesign.model.rdesign import RNAModel
# from rdesign.utils.data import RNADataModule
# from rdesign.utils.train import get_trainer

# model = RNAModel()
# data = RNADataModule.from_defaults(batch_size=8, split_ratio=[0.95, 0.05, 0.0])
# trainer = get_trainer(name='RDesign', version=2.5, max_epochs=80)
# trainer.fit(model, data)

from rnampnn.utils.data import RNADataModule
from rnampnn.utils.train import get_trainer
from rnampnn.model.rnampnn import RNAMPNN
from rnampnn.config.seeds import seeding

seeding()
model = RNAMPNN()
data = RNADataModule(split_ratio=0.95, batch_size=2, min_len=2)
trainer = get_trainer(name='RNAMPNN-AF', version=1, max_epochs=120)
trainer.fit(model, data)


# model = RNAMPNN.load_from_checkpoint('out/checkpoints/RNAMPNN/checkpoint-epoch=59-6.ckpt')
#
# data = RNADataModule(split_ratio=0.999, batch_size=1, min_len=2)
# data.setup()
# for batch in data.train_dataloader():
#     model.eval()
#     sequence, coords, mask, id = batch
#     coords  = coords.to(model.device)
#     mask = mask.to(model.device)
#     x = model(coords, mask)