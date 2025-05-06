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

model = RNAMPNN()
data = RNADataModule(split_ratio=0.95, batch_size=1)
trainer = get_trainer(name='RNAMPNN', version=0, max_epochs=80)
trainer.fit(model, data)