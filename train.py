from rnampnn.utils.data import RNADataModule
from rnampnn.utils.train import get_trainer
from rnampnn.model.rnampnn import RNAMPNN
from rnampnn.config.seeds import seeding

seeding()
model = RNAMPNN()
data = RNADataModule(split_ratio=0.95, batch_size=2)
trainer = get_trainer(name='RNAMPNN-AF', version=2, max_epochs=90)
trainer.fit(model, data)