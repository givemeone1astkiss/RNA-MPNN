from rnampnn.utils import gen_dataframe
from rnampnn.config import COMPETITION_DATA, COMPETITION_OUT
from rnampnn.utils.data import RNADataset, featurize
from rnampnn.model import RNAModel
import torch
from tqdm import tqdm


def predict(model: RNAModel, batch_size=32, data_path=COMPETITION_DATA, output_path=COMPETITION_OUT):
    
    gen_dataframe(file_path=f'{data_path}seqs/').to_csv(f'{data_path}predict_data.csv', index=False)
    predict_dataset = RNADataset(f'{data_path}predict_data.csv', f'{data_path}coords')
    data = torch.utils.data.DataLoader(predict_dataset, batch_size, num_workers=19, shuffle=False, persistent_workers=True, collate_fn=featurize)
    for batch_id, batch in tqdm(enumerate(data), total=len(data), desc="Predicting", unit="batch"):
        model.predict(batch, batch_id, output_dir=output_path)
    print(f"Predictions saved to {output_path}")