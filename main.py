from rnampnn import *

if __name__ == "__main__":
    # Load the model
    seeding()
    model = RNAModel.load_from_checkpoint('out/checkpoints/RDesign/checkpoint-epoch=59-1.ckpt')

    # Predict using the model
    try:
        predict(model, batch_size=4)
    except torch.OutOfMemoryError:
        predict(model, batch_size=2)
    finally:
        predict(model, batch_size=1)