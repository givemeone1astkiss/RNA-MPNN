from rnampnn.model.rdesign import RNAModel
from rnampnn.config.glob import BEST_CKPT
from rnampnn.config.seeds import seeding
from rnampnn.utils.predict import predict
import torch
import argparse


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict RNA structure")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for prediction")
    args = parser.parse_args()

    # Load the model
    seeding()
    model = RNAModel.load_from_checkpoint(BEST_CKPT)

    # Predict using the model
    try:
        predict(model, batch_size=args.batch_size)
    except torch.OutOfMemoryError:
        predict(model, batch_size=2)
    finally:
        predict(model, batch_size=1)