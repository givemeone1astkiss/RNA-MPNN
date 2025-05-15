from rnampnn.utils.data import RNADataModule
from rnampnn.utils.train import get_trainer
from rnampnn.model.rnampnn import RNAMPNN
from rnampnn.utils.seed import seeding

seeding()
model = RNAMPNN()
data = RNADataModule(split_ratio=0.95, batch_size=3)
trainer = get_trainer(name='RNAMPNN-AF', version=7, max_epochs=200)
trainer.fit(model, data)