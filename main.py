from rnampnn.config.seeds import seeding
from rnampnn.model.rnampnn import RNAMPNN
from rnampnn.config.glob import BEST_CKPT
from rnampnn.utils.predict import predict
import torch


if __name__ == "__main__":
    seeding()
    model = RNAMPNN.load_from_checkpoint(BEST_CKPT)

    # Predict using the model
    try:
        predict(model, batch_size=2)
    except torch.OutOfMemoryError:
        predict(model, batch_size=1)