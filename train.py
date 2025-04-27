from rnampnn.model.rdesign import RNAModel
from rnampnn.utils.data import RNADataModule
from rnampnn.utils.train import get_trainer

model = RNAModel()
data = RNADataModule.from_defaults(batch_size=8, split_ratio=[0.95, 0.05, 0.0])
trainer = get_trainer(name='RDesign', version=2.5, max_epochs=80)
trainer.fit(model, data)