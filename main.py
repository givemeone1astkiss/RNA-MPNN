# from rnampnn.utils.seed import seeding
# from rnampnn.model.rnampnn import RNAMPNN
# from rnampnn.config.glob import BEST_CKPT, DATA_PATH, OUTPUT_PATH
# from rnampnn.utils.predict import predict
# import torch
#
#
# if __name__ == "__main__":
#     seeding()
#     model = RNAMPNN.load_from_checkpoint(BEST_CKPT)
#
#     # Predict using the model
#     try:
#         predict(model, batch_size=2)
#     except torch.OutOfMemoryError:
#         predict(model, batch_size=1)

from rdesign.model.rdesign import RNAModel
from rdesign.config.glob import BEST_CKPT, DATA_PATH, OUTPUT_PATH
from rdesign.config.seeds import seeding
from rdesign.utils.predict import predict
import torch


if __name__ == "__main__":
    seeding()
    model = RNAModel.load_from_checkpoint(BEST_CKPT)

    try:
        predict(model, batch_size=4)
    except torch.OutOfMemoryError:
        predict(model, batch_size=2)