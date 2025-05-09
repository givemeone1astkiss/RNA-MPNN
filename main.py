# from rdesign.model.rdesign import RNAModel
# from rdesign.config.glob import BEST_CKPT
# from rdesign.config.seeds import seeding
# from rdesign.utils.predict import predict
# import torch
# import argparse


# if __name__ == "__main__":
#     seeding()
#     model = RNAModel.load_from_checkpoint(BEST_CKPT)

#     # Predict using the model
#     try:
#         predict(model, batch_size=4)
#     except torch.OutOfMemoryError:
#         predict(model, batch_size=2)
#     finally:
#         predict(model, batch_size=1)
        
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
        predict(model, batch_size=1)
    except torch.OutOfMemoryError:
        print("Out of memory error, trying with batch size 1")