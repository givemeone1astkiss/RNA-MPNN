from rnampnn import *

model = RNAModel()
data = RNADataModule.from_defaults(batch_size=8)
trainer = get_trainer(name='RDesign', version=1, max_epochs=60)
trainer.fit(model, data)