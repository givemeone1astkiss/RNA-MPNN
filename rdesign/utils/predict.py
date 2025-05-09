import pandas as pd
from ..config.glob import COMPETITION_DATA, COMPETITION_OUT
from ..utils.data import RNADataset, featurize, gen_dataframe
from ..model.rdesign import RNAModel
import torch
from tqdm import tqdm
import os


def predict(model: RNAModel, batch_size=32, data_path=COMPETITION_DATA, output_path=COMPETITION_OUT, filename="submit.csv")->None:
    """
    Predicts the output of the model using the provided data.

    Args:
        model (RNAModel): The trained model to use for predictions.
        batch_size (int): The batch size to use for predictions.
        data_path (str): The path to the data directory.
        output_path (str): The path to the output directory.
        filename (str): The name of the output file.
    """
    gen_dataframe(file_path=f'{data_path}seqs/').to_csv('./predict_data.csv', index=False)
    predict_dataset = RNADataset('./predict_data.csv', f'{data_path}coords')
    data = torch.utils.data.DataLoader(predict_dataset, batch_size, num_workers=19, shuffle=False, persistent_workers=True, collate_fn=featurize)
    for batch_id, batch in tqdm(enumerate(data), total=len(data), desc="Predicting", unit="batch", position=0):
        model.predict(batch, batch_id, output_dir='./', filename=filename)
    result = pd.read_csv(f"./{filename}")
    result.to_csv(f"{output_path}{filename}", index=False)
    os.remove('./predict_data.csv')
    os.remove(f'./{filename}')
    print(f"Predictions saved to {output_path}{filename}")