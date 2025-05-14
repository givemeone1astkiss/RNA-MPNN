from rnampnn.utils.data import RNADataModule
from rnampnn.utils.train import get_trainer
from rnampnn.model.rnampnn import RNAMPNN
from rnampnn.utils.seed import seeding

seeding()
model = RNAMPNN()
data = RNADataModule(split_ratio=0.9, batch_size=3, noise_augmentation=200)
trainer = get_trainer(name='RNAMPNN-AF', version=4, max_epochs=600)
trainer.fit(model, data)