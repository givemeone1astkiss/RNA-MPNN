from typing import final
from Bio import SeqIO
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from ..config.glob import DATA_PATH, SPLIT_RATIO
import torch
from matplotlib import pyplot as plt
import seaborn as sns

def read_fasta_biopython(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def gen_dataframe(file_path=DATA_PATH+'seqs/'):
    train_file_list = os.listdir(file_path)
    content_dict = {
        "pdb_id": [],
        "seq": []
    }

    for file in tqdm(train_file_list, desc="Reading files", unit="file"):
        sequences = read_fasta_biopython(file_path + file)
        try:
            content_dict["pdb_id"].append(list(sequences.keys())[0])
        except IndexError:
            print(f"Error reading file {file}: {sequences}")
            continue
        try:
            content_dict["seq"].append(list(sequences.values())[0])
        except IndexError:
            print(f"Error reading file {file}: {sequences}")
            continue

    return pd.DataFrame(content_dict)

def split_dataset(data, output_dir=DATA_PATH, split_ratio=SPLIT_RATIO):

    split = np.random.choice(['train', 'valid', 'test'], size=len(data), p=split_ratio)
    data['split'] = split
    train_data = data[data['split'] == 'train']
    valid_data = data[data['split'] == 'valid']
    test_data = data[data['split'] == 'test']
    train_data.to_csv(output_dir + "train_data.csv", index=False)
    valid_data.to_csv(output_dir + "valid_data.csv", index=False)
    test_data.to_csv(output_dir + "test_data.csv", index=False)

class RNADataset(Dataset):
    def __init__(self, data_path, npy_dir):
        super(RNADataset, self).__init__()
        self.data = pd.read_csv(data_path)
        self.npy_dir = npy_dir
        self.seq_list = self.data['seq'].to_list()
        self.name_list = self.data['pdb_id'].to_list()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        pdb_id = self.name_list[idx]
        coords = np.load(os.path.join(self.npy_dir, pdb_id + '.npy'))

        feature = {
            "name": pdb_id,
            "seq": seq,
            "coords": {
                "P": coords[:, 0, :],
                "O5'": coords[:, 1, :],
                "C5'": coords[:, 2, :],
                "C4'": coords[:, 3, :],
                "C3'": coords[:, 4, :],
                "O3'": coords[:, 5, :],
            }
        }

        return feature

def featurize(batch):
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    names = []

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([np.nan_to_num(b['coords'][c], nan=0.0) for c in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]], 1)
        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
        X[i, :, :, :] = x_pad
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        names.append(b['name'])

    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X) + np.nan
    for i, n in enumerate(numbers):
        X_new[i, :n, ::] = X[i][mask[i] == 1]
        S_new[i, :n] = S[i][mask[i] == 1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, mask, lengths, names

@final
class RNADataModule(LightningDataModule):

    @classmethod
    def from_defaults(cls, batch_size=16, split_ratio=SPLIT_RATIO):
        split_dataset(gen_dataframe(), split_ratio=split_ratio)
        return cls(
            train_data_path=DATA_PATH + 'train_data.csv',
            valid_data_path=DATA_PATH + 'valid_data.csv',
            test_data_path=DATA_PATH + 'test_data.csv',
            train_npy_dir='./data/coords',
            valid_npy_dir='./data/coords',
            test_npy_dir='./data/coords',
            batch_size=batch_size
        )

    def __init__(self, train_data_path, valid_data_path, test_data_path, train_npy_dir, valid_npy_dir, test_npy_dir, batch_size=32):
        super().__init__()
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path
        self.train_npy_dir = train_npy_dir
        self.valid_npy_dir = valid_npy_dir
        self.test_npy_dir = test_npy_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = RNADataset(self.train_data_path, self.train_npy_dir)
            self.valid_dataset = RNADataset(self.valid_data_path, self.valid_npy_dir)

        if stage == 'test' or stage is None:
            self.test_dataset = RNADataset(self.test_data_path, self.test_npy_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=19, shuffle=True, persistent_workers=True, collate_fn=featurize)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=19, shuffle=False, persistent_workers=True, collate_fn=featurize)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=19, shuffle=False, persistent_workers=True, collate_fn=featurize)

def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def gen_seq_csv(data_path: str=f'{DATA_PATH}seqs/', output_path: str=f'{DATA_PATH}ref.csv'):
    content_dict = {
        "pdb_id": [],
        "seq": []
    }

    fasta_files = os.listdir(data_path)

    for file in tqdm(fasta_files, desc="Reading files", unit="file"):
        file_path = os.path.join(data_path, file)
        try:
            for record in SeqIO.parse(file_path, "fasta"):
                content_dict["pdb_id"].append(record.id)
                content_dict["seq"].append(str(record.seq))
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    df = pd.DataFrame(content_dict)
    df.to_csv(output_path, index=False)
    print(f"CSV file saved at {output_path}")

def cal_recovery_rate(pred_path: str, ref_path: str, output_path: str=f'{DATA_PATH}recovery.csv'):
    pred_df = pd.read_csv(pred_path)
    ref_df = pd.read_csv(ref_path)

    merged_df = pd.merge(ref_df, pred_df, on='pdb_id', suffixes=('_ref', '_pred'))

    recovery_data = []
    for _, row in tqdm(merged_df.iterrows(), desc="Calculating recovery rates", total=len(merged_df)):
        ref_seq = row['seq_ref']
        pred_seq = row['seq_pred']
        length = len(ref_seq)
        recovery_rate = sum(1 for r, p in zip(ref_seq, pred_seq) if r == p) / length
        recovery_data.append({
            'pdb_id': row['pdb_id'],
            'recovery_rate': recovery_rate,
            'length': length
        })

    recovery_df = pd.DataFrame(recovery_data)
    recovery_df.to_csv(output_path, index=False)
    print(f"Recovery rates saved at {output_path}")

def draw_recovery_scatter(recovery_path: str, output_path: str=f'{DATA_PATH}recovery_scatter.png'):
    recovery_df = pd.read_csv(recovery_path)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=recovery_df, x='length', y='recovery_rate')
    plt.title('Recovery Rate vs Length')
    plt.xlabel('Length')
    plt.ylabel('Recovery Rate')
    plt.savefig(output_path)
    print(f"Scatter plot saved at {output_path}")

def separate(concat: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(lengths.max().item())
    if len(concat.shape) == 1:
        separated = torch.zeros((batch_size, max_length), dtype=concat.dtype)
    if len(concat.shape) == 2:
        separated = torch.zeros((batch_size, max_length, concat.shape[1]), dtype=concat.dtype)

    start_idx = 0
    for i, length in enumerate(lengths):
        end_idx = start_idx + int(length.item())
        separated[i, :int(length.item())] = concat[start_idx:end_idx]
        start_idx = end_idx

    return separated

def concat(separated: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(lengths.max().item())
    if len(separated.shape) == 2:
        concatenated = torch.zeros((batch_size, max_length), dtype=separated.dtype)
    if len(separated.shape) == 3:
        concatenated = torch.zeros((batch_size, max_length, separated.shape[2]), dtype=separated.dtype)

    start_idx = 0
    for i, length in enumerate(lengths):
        end_idx = start_idx + int(length.item())
        concatenated[start_idx:end_idx] = separated[i, :int(length.item())]
        start_idx = end_idx

    return concatenated


def gen_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(lengths.max().item())
    mask = torch.zeros((batch_size, max_length), dtype=torch.float32)
    for i, length in enumerate(lengths):
        mask[i, :int(length.item())] = 1.0
    return mask