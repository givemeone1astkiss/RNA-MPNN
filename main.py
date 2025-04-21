from rnampnn import *

if __name__ == "__main__":
    # Load the model
    model = RNAModel.load_from_checkpoint('out/checkpoints/RDesign/checkpoint-epoch=29.ckpt')

    # Predict using the model
    predict(model, batch_size=8)