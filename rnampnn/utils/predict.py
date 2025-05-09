import pandas as pd
from .data import analyse_dataset
from ..config.glob import COMPETITION_DATA, COMPETITION_OUT
from ..utils.data import RNADataset, featurize
from ..model.rnampnn import RNAMPNN
import torch
from tqdm import tqdm
import os


def predict(model: RNAMPNN, batch_size=32, data_path=COMPETITION_DATA, output_path=COMPETITION_OUT, filename="submit.csv")->None:
    """
    Predicts the output of the model using the provided data.

    Args:
        model (RNAMPNN): The trained model to use for predictions.
        batch_size (int): The batch size to use for predictions.
        data_path (str): The path to the data directory.
        output_path (str): The path to the output directory.
        filename (str): The name of the output file.
    """
    predict_dataset = RNADataset.from_path(data_path=data_path, is_predict=True)
    data = torch.utils.data.DataLoader(predict_dataset, batch_size, num_workers=19, shuffle=False, persistent_workers=True, collate_fn=featurize)
    for batch_id, batch in tqdm(enumerate(data), total=len(data), desc="Predicting", unit="batch", position=0):
        model.predict(batch, batch_id, output_dir='./', filename=filename)
    result = pd.read_csv(f"./{filename}")
    result.to_csv(f"{output_path}{filename}", index=False)
    os.remove(f'./{filename}')
    print(f"Predictions saved to {output_path}{filename}")