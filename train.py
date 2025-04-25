from rnampnn.model.rdesign import RNAModel
from rnampnn.utils.data import RNADataModule
from rnampnn.utils.train import get_trainer

model = RNAModel()
data = RNADataModule.from_defaults(batch_size=8)
trainer = get_trainer(name='RDesign', version=1, max_epochs=60)
trainer.fit(model, data)