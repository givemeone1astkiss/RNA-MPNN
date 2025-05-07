from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union, Tuple
from Bio import SeqIO
from Bio.PDB import PDBParser, is_aa
import numpy as np
from ..config.glob import DATA_PATH, MIN_LEN, VOCAB, NUM_MAIN_SEQ_ATOMS, NUM_RES_TYPES
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import torch


def analyse_dataset(path=f'{DATA_PATH}seqs/'):
    """
    Analyse a dataset of fasta files in the given directory.

    Args:
        path (str): Path to the directory containing fasta files.

    Prints:
        - Longest sequence length and its corresponding ID.
        - Shortest sequence length and its corresponding ID.
        - Average sequence length.
        - Median sequence length.
    Outputs:
        - A histogram of sequence length distribution.
    """
    longest_id, longest_length = None, 0
    shortest_id, shortest_length = None, float('inf')
    sequence_lengths = []

    # Iterate through all fasta files in the directory
    for file in tqdm(os.listdir(path), desc="Processing fasta files"):
        if file.endswith(".fasta") or file.endswith(".fa"):
            file_path = os.path.join(path, file)
            for record in SeqIO.parse(file_path, "fasta"):
                seq_length = len(record.seq)
                sequence_lengths.append(seq_length)
                if seq_length > longest_length:
                    longest_length = seq_length
                    longest_id = record.id
                if seq_length < shortest_length:
                    shortest_length = seq_length
                    shortest_id = record.id

    # Calculate average and median sequence lengths
    average_length = np.mean(sequence_lengths)
    median_length = np.median(sequence_lengths)

    print(f"Longest sequence: ID = {longest_id}, Length = {longest_length}")
    print(f"Shortest sequence: ID = {shortest_id}, Length = {shortest_length}")
    print(f"Average sequence length: {average_length:.2f}")
    print(f"Median sequence length: {median_length:.2f}")

    # Plot the sequence length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def pdb_to_coords(input_path: str, output_path: str) -> None:
    """
    Extracts coordinates of specific atoms from RNA PDB files and saves them as .npy files.

    Args:
        input_path (str): Path to the directory containing RNA PDB files.
        output_path (str): Path to the directory where .npy files will be saved.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    parser = PDBParser(QUIET=True)
    atom_names = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "N1", "N9"]

    for pdb_file in os.listdir(input_path):
        if not pdb_file.endswith(".pdb"):
            continue

        pdb_path = os.path.join(input_path, pdb_file)
        structure = parser.get_structure(pdb_file, pdb_path)

        coords_list = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if not is_aa(residue, standard=False):
                        residue_coords = []
                        for atom_name in atom_names:
                            if atom_name in residue:
                                atom = residue[atom_name]
                                residue_coords.append(atom.coord)
                            else:
                                residue_coords.append([np.nan, np.nan, np.nan])
                        coords_list.append(residue_coords)

        coords_array = np.array(coords_list, dtype=np.float32)

        output_file = os.path.join(output_path, os.path.splitext(pdb_file)[0] + ".npy")
        np.save(output_file, coords_array)

class RNADataset(Dataset):
    def __init__(self, data: List[Dict[str, Union[str, torch.Tensor]]]) -> None:
        """
        Initialize the RNA dataset with a list of dictionaries.

        Args:
            data (List[Dict[str, Union[str, torch.Tensor]]]): List of dictionaries with keys: 'id', 'sequence', 'coordinates'.
        """
        super().__init__()
        self.data = data

    @classmethod
    def from_path(cls, data_path: str, is_predict: bool = False) -> "RNADataset":
        """
        Create an RNADataset instance by reading data from the specified directory.

        Args:
            data_path (str): Path to the directory containing `seqs/` and `coords/` subdirectories.
            is_predict (bool): If True, generate dummy sequences instead of reading from `seqs/`.

        Returns:
            RNADataset: An instance of RNADataset initialized with the loaded data.
        """
        data = []
        seqs_path = os.path.join(data_path, "seqs")
        coords_path = os.path.join(data_path, "coords")

        npy_files = [file for file in os.listdir(coords_path) if file.endswith(".npy")]
        for file in tqdm(npy_files, desc="Initializing dataset"):
            file_path = os.path.join(coords_path, file)
            coordinates = torch.tensor(np.load(file_path), dtype=torch.float32)
            if torch.isnan(coordinates).any():
                coordinates = cls.__fill_nan_with_mean(coordinates)
            rna_id = os.path.splitext(file)[0]

            if is_predict:
                dummy_seq = str("A" * coordinates.shape[0])
                sequence = cls._one_hot_encode(dummy_seq)
            else:
                sequence = cls._load_sequence(seqs_path, rna_id)

            data.append({'id': rna_id, 'sequence': sequence, 'coordinates': coordinates})

        return cls(data)

    @staticmethod
    def __fill_nan_with_mean(coordinates: torch.Tensor) -> torch.Tensor:
        """
        Fill NaN values in the coordinates tensor with random values based on the coordinates of other atoms.
        If NaN values still exist after processing, they will be replaced with 0.

        Args:
            coordinates (torch.Tensor): Tensor of shape (num_samples, num_atoms, 3) containing coordinates.

        Returns:
            torch.Tensor: Tensor with NaN values filled with random coordinates or 0.
        """
        coords_np = coordinates.numpy()
        for atom_idx in range(coords_np.shape[1]):
            atom_coords = coords_np[:, atom_idx, :]
            nan_mask = np.isnan(atom_coords).any(axis=1)
            for seq_idx in np.where(nan_mask)[0]:
                if atom_idx < 6:
                    non_nan_indices = np.where(~np.isnan(coords_np[seq_idx, :, :]).any(axis=1))[0]
                    if len(non_nan_indices) > 0:
                        reference_atom_idx = non_nan_indices[0]
                        reference_coords = coords_np[seq_idx, reference_atom_idx, :]
                        random_vector = np.random.randn(3)
                        random_vector = 1.5 * random_vector / np.linalg.norm(random_vector)
                        atom_coords[seq_idx] = reference_coords + random_vector
                elif atom_idx == 6:
                    reference_coords = coords_np[seq_idx, 5, :]
                    if not np.isnan(reference_coords).any():
                        random_vector = np.random.randn(3)
                        random_vector = NUM_RES_TYPES * random_vector / np.linalg.norm(random_vector)
                        atom_coords[seq_idx] = reference_coords + random_vector

        coords_np[np.isnan(coords_np)] = 0

        return torch.tensor(coords_np, dtype=torch.float32)

    @staticmethod
    def _one_hot_encode(sequence: str) -> torch.Tensor:
        """
        Convert a nucleotide sequence into one-hot encoding.

        Args:
            sequence (str): RNA sequence consisting of 'A', 'U', 'C', 'G'.

        Returns:
            torch.Tensor: One-hot encoded representation of the sequence.
        """
        one_hot = np.zeros((len(sequence), NUM_RES_TYPES), dtype=np.float32)
        for i, nucleotide in enumerate(sequence):
            if nucleotide in VOCAB:
                one_hot[i, VOCAB[nucleotide]] = 1.0
        return torch.tensor(one_hot, dtype=torch.float32)

    @staticmethod
    def _load_sequence(seqs_path: str, rna_id: str) -> torch.Tensor:
        """
        Load the RNA sequence from a fasta file and convert it to one-hot encoding.

        Args:
            seqs_path (str): Path to the directory containing fasta files.
            rna_id (str): RNA ID to look for in the fasta file.

        Returns:
            torch.Tensor: One-hot encoded sequence tensor.
        """
        fasta_file = os.path.join(seqs_path, f"{rna_id}.fasta")
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"Sequence file for RNA ID {rna_id} not found.")
        for record in SeqIO.parse(fasta_file, "fasta"):
            return RNADataset._one_hot_encode(str(record.seq))
        raise ValueError(f"No valid sequence found in {fasta_file}.")

    def filter_by_min_length(self, min_len: int) -> None:
        """
        Remove all data points from the dataset where the sequence length is less than `min_len`.

        Args:
            min_len (int): Minimum sequence length to retain in the dataset.
        """
        self.data = [item for item in self.data if item['sequence'].shape[0] >= min_len]

    def noise_augmentation(self, num_gen: int) -> None:
        """
        Sample `num_gen` data points with replacement from the dataset and add Gaussian noise
        to their coordinates. The noise has a mean of 0 and a variance of 1e-2. The RNA ID
        and reference sequence remain unchanged.

        Args:
            num_gen (int): Number of samples to generate with noise.
        """
        for _ in tqdm(range(num_gen), desc="Generating noisy samples"):
            idx: int = np.random.randint(len(self.data))
            sample: Dict[str, Union[str, torch.Tensor]] = self.data[idx]

            noise: torch.Tensor = torch.normal(mean=0, std=1e-2, size=sample['coordinates'].shape)
            noisy_coordinates: torch.Tensor = sample['coordinates'] + noise

            self.data.append(
                {'id': sample['id'], 'sequence': sample['sequence'], 'coordinates': noisy_coordinates})

    def slice_augmentation(self, num_gen: int, min_len: int=MIN_LEN) -> None:
        """
        Generate new data points by slicing sequences and coordinates from existing data.

        Args:
            num_gen (int): Number of new data points to generate.
            min_len (int): Minimum length of the sequence to consider for slicing.
        """
        for _ in tqdm(range(num_gen), desc="Generating sliced samples"):
            valid_data = [item for item in self.data if item['sequence'].shape[0] > min_len]
            if not valid_data:
                raise ValueError("No sequences longer than min_len available for slicing.")

            sample = random.choice(valid_data)
            seq_len = sample['sequence'].shape[0]

            start_idx = random.randint(0, seq_len - min_len)

            sliced_sequence = sample['sequence'][start_idx:start_idx + min_len]
            sliced_coordinates = sample['coordinates'][start_idx:start_idx + min_len]

            new_data_point = {
                'id': sample['id'],
                'sequence': sliced_sequence,
                'coordinates': sliced_coordinates
            }

            self.data.append(new_data_point)

    def getitem_by_key(self, target_id: str) -> List[Dict[str, Union[str, torch.Tensor]]]:
        """
        Retrieve all data points in the dataset where the RNA ID matches the given `id`.

        Args:
            target_id (str): The RNA ID to query.

        Returns:
            list: A list of dictionaries containing 'id', 'sequence', and 'coordinates' for all matching data points.
        """
        return [item for item in self.data if item['id'] == target_id]

    def get_ids(self) -> List[str]:
        """
        Retrieve all unique RNA IDs in the dataset.

        Returns:
            list: A list of unique RNA IDs.
        """
        return list(set(item['id'] for item in self.data))

    def shuffle(
            self,
            return_perm: bool = False,
    ) -> Union['Dataset', Tuple['Dataset', torch.Tensor]]:
        """
        Shuffle the dataset in place and return the shuffled dataset.

        Args:
            return_perm (bool): If True, return the permutation indices used for shuffling.

        Returns:
            RNADataset: The shuffled dataset.
            Tensor: Permutation indices if return_perm is True.
        """
        perm = torch.randperm(len(self.data))
        self.data = [self.data[i] for i in perm]
        if return_perm:
            return self, perm
        return self

    def __len__(self) -> int:
        """
        Return the number of RNA samples in the dataset.

        Returns:
            int: Number of RNA samples.
        """
        return len(self.data)

    def len(self) -> int:
        """
        Return the number of RNA samples in the dataset.

        Returns:
            int: Number of RNA samples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Get the sequence, coordinates, and RNA ID at the given index.

        Args:
            idx (int): Index of the RNA sample.

        Returns:
            dict: A dictionary containing 'id', 'sequence', and 'coordinates'.
        """
        return self.data[idx]

class RNADataModule(LightningDataModule):
    def __init__(self, data_path=DATA_PATH, split_ratio: float|List[float] = 0.9, batch_size: int=8, noise_augmentation: int|None=None, slice_augmentation: int|None=None, min_len: int|None=None) -> None:
        """
        Initialize the RNA data module.
        Args:
            data_path (str): Path to the dataset directory.
            split_ratio (float|List[float]): Ratio for splitting the dataset into train and validation sets.
                If a float, it represents the proportion of the training set. If a list, it represents
                the proportions for multiple splits.
            batch_size (int): Target sum of RNA sequence lengths for each batch.
            noise_augmentation (int|None): Number of noisy samples to generate for data augmentation.
            slice_augmentation (int|None): Number of sliced samples to generate for data augmentation.
            min_len (int|None): Minimum sequence length to retain in the dataset.
        """
        super().__init__()
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.min_len = min_len
        self.batch_size = batch_size
        self.noise_augmentation = noise_augmentation
        self.slice_augmentation = slice_augmentation
        self.train_set = None
        self.val_set = None
        self.test_set = None


    def setup(self, stage: str|None=None) -> None:
        if stage == "fit" or stage is None:
            assert isinstance(self.split_ratio, float) and self.split_ratio > 0, "Invalid split ratio. Must be a float between 0 and 1."
            raw = RNADataset.from_path(self.data_path)
            if self.noise_augmentation is not None:
                raw.noise_augmentation(num_gen=self.noise_augmentation)
            if self.slice_augmentation is not None:
                raw.slice_augmentation(num_gen=self.slice_augmentation)
            if self.min_len is not None:
                raw.filter_by_min_length(min_len=self.min_len)
            self.train_set, self.val_set = self._split_dataset(raw,split_ratio=self.split_ratio)
        if stage == "test" or stage is None:
            self.test_set = RNADataset.from_path(self.data_path, is_predict=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=19, shuffle=True, persistent_workers=True, collate_fn=self._featurize)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=19, shuffle=False, persistent_workers=True, collate_fn=self._featurize)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=19, shuffle=False, persistent_workers=True, collate_fn=self._featurize)

    @staticmethod
    def _featurize(batch: List[Dict[str, Union[str, torch.Tensor]]]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Featurize a batch of RNA data by padding sequences and coordinates to the same length.

        Args:
            batch (List[Dict[str, Union[str, torch.Tensor]]]): A batch of RNA data points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
                - sequences: Padded one-hot encoded sequences of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS).
                - coords: Padded coordinates of atoms of shape (batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3).
                - mask: Mask indicating valid positions in the sequences of shape (batch_size, max_len).
                - rna_ids: List of RNA IDs corresponding to the batch.
        """

        batch_size = len(batch)
        max_len = max(item['sequence'].shape[0] for item in batch)
        sequences = torch.zeros((batch_size, max_len, NUM_RES_TYPES), dtype=torch.float32)
        coords = torch.zeros((batch_size, max_len, NUM_MAIN_SEQ_ATOMS, 3), dtype=torch.float32)
        mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

        rna_ids = []


        for i, item in enumerate(batch):
            seq_len = item['sequence'].shape[0]
            sequences[i, :seq_len, :] = item['sequence']
            coords[i, :seq_len, :, :] = item['coordinates']
            mask[i, :seq_len] = 1
            rna_ids.append(item['id'])

        return sequences, coords, mask, rna_ids

    @staticmethod
    def _split_dataset(dataset: RNADataset, split_ratio: Union[float, List[float]]) -> Tuple[RNADataset, ...]:
        """
        Split an RNADataset object into multiple subsets based on the given split_ratio.

        Args:
            dataset (RNADataset): The dataset to split.
            split_ratio (Union[float, List[float]]): A float (for train-test split) or a list of floats
                (for multiple subsets). The sum of the list must equal 1.

        Returns:
            Tuple[RNADataset, ...]: A tuple of RNADataset objects representing the subsets.
        """
        # Validate split_ratio
        if isinstance(split_ratio, float):
            if not (0 < split_ratio < 1):
                raise ValueError("Proportion must be a float between 0 and 1.")
            split_ratio = [split_ratio, 1 - split_ratio]
        elif isinstance(split_ratio, list):
            if not all(isinstance(p, float) and p > 0 for p in split_ratio):
                raise ValueError("All split_ratio must be positive floats.")
            if not abs(sum(split_ratio) - 1) < 1e-6:
                raise ValueError("The sum of split_ratio must equal 1.")
        else:
            raise TypeError("split_ratio must be a float or a list of floats.")

        # Group data points by their 'id'
        grouped_data = {}
        for item in dataset.data:
            grouped_data.setdefault(item['id'], []).append(item)

        # Shuffle the grouped data
        grouped_ids = list(grouped_data.keys())
        random.shuffle(grouped_ids)

        # Calculate the sizes of each subset
        total_groups = len(grouped_ids)
        subset_sizes = [int(total_groups * p) for p in split_ratio]
        subset_sizes[-1] += total_groups - sum(subset_sizes)

        # Split the grouped data into subsets
        subsets = []
        start_idx = 0
        for size in subset_sizes:
            subset_ids = grouped_ids[start_idx:start_idx + size]
            subset_data = [item for rna_id in subset_ids for item in grouped_data[rna_id]]
            subsets.append(RNADataset(subset_data))
            start_idx += size

        return tuple(subsets)