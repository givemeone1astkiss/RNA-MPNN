import pytorch_lightning as pl
import numpy as np

from ..config.glob import OUTPUT_PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback
from ..model.rdesign import RNAModel
import torch
from tqdm import tqdm
import pickle

class NameModel(Callback):
    def __init__(self, name: str, version: int):
        super().__init__()
        self.name = name
        self.version = version

    def on_fit_start(self, trainer: pl.Trainer, model: RNAModel) -> None:
        model.name = self.name
        model.version = self.version


class LossMonitor(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, model: RNAModel):
        avg_loss = torch.tensor([x['val_loss'] * x['len'] for x in model.val_step_outputs]).sum(dim=-1).to(device=model.device) / torch.tensor(
            [x['len'] for x in model.val_step_outputs]).sum(dim=-1).to(device=model.device)
        avg_recovery_rate = torch.tensor([x['correct'] for x in model.val_step_outputs]).sum(dim=-1).to(device=model.device) / torch.tensor(
            [x['len'] for x in model.val_step_outputs]).sum(dim=-1).to(device=model.device)

        model.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        model.log('val_recovery_rate', avg_recovery_rate, prog_bar=True, sync_dist=True)
        model.val_step_outputs = []

    def on_test_epoch_end(self, trainer: pl.Trainer, model: RNAModel):
        avg_loss = torch.tensor([x['test_loss'] * x['len'] for x in model.test_step_outputs]).sum(dim=-1).to(device=model.device) / torch.tensor(
            [x['len'] for x in model.test_step_outputs]).sum(dim=-1).to(device=model.device)
        avg_recovery_rate = torch.tensor([x['correct'] for x in model.test_step_outputs]).sum(dim=-1).to(device=model.device) / torch.tensor(
            [x['len'] for x in model.test_step_outputs]).sum(dim=-1).to(device=model.device)

        model.log('test_loss', avg_loss, prog_bar=True, sync_dist=True)
        model.log('test_recovery_rate', avg_recovery_rate, prog_bar=True, sync_dist=True)
        model.test_step_outputs = []

class XGBTrainer(Callback):
    def __init__(self):
        super().__init__()
        self.batch_val_loss = []
        self.batch_val_length = []
        self.batch_val_correct = []

    def on_fit_end(self, trainer: pl.Trainer, model: RNAModel) -> None:
        print('=' * 20, '\n')
        print('Start training XGBoost!\n')
        X, Y = self._generate_embedding(trainer.train_dataloader, model)
        model.xgb_readout.fit(X, Y)
        train_score = model.xgb_readout.score(X, Y)
        print(f'Training score: {train_score}\n')
        print('Start validation!\n')
        X, Y = self._generate_embedding(trainer.val_dataloaders, model)
        val_score = model.xgb_readout.score(X, Y)
        print(f'Validation score: {val_score}\n')
        print('XGBoost training done!')
        print('=' * 20, '\n')
        with open(f"{OUTPUT_PATH}/checkpoints/{model.name}/XGB-V{model.version}.pkl", 'wb') as f:
            pickle.dump(model.xgb_readout, f)
        trainer.save_checkpoint(f"{OUTPUT_PATH}/checkpoints/{model.name}/Final-V{model.version}.ckpt")

    @staticmethod
    def _generate_embedding(dataloader: torch.utils.data.DataLoader, model: RNAModel):
        embeddings = np.ndarray((0, model.hparams.hidden_dim))
        sequences = np.ndarray(0)
        for batch in tqdm(dataloader, desc="Generating Embedding...", total=len(dataloader), position=0):
            X, S, mask, lengths, _ = batch
            X = X.to(model.device)
            S = S.to(model.device)
            mask = mask.to(model.device)
            h_V, S = model(X, S, mask)
            embeddings = np.append(embeddings, h_V.to(device=torch.device(torch.device('cpu'))).detach().numpy(), axis=0)
            sequences = np.append(sequences, S.to(device=torch.device(torch.device('cpu'))).detach().numpy(), axis=0)

        return embeddings, sequences

def get_trainer(name: str, version: int, max_epochs: int=30, val_check_interval: int = 1):
    logger = pl.loggers.TensorBoardLogger(
        save_dir=f"{OUTPUT_PATH}/logs",
        name=name,
        version=version,
    )

    checkpoint = ModelCheckpoint(
        dirpath=f"{OUTPUT_PATH}/checkpoints/{name}/",
        filename='checkpoint-{epoch:02d}'+f'-{version}',
        save_top_k=1,
        verbose=True,
        monitor='val_recovery_rate',
        mode='max',
    )

    return pl.Trainer(
        accelerator="auto",
        devices=-1 if torch.cuda.is_available() else 1,
        precision="bf16-mixed",
        max_epochs=max_epochs,
        enable_progress_bar=True,
        logger=logger,
        callbacks=[checkpoint, LossMonitor(), XGBTrainer(), NameModel(name, version)],
        check_val_every_n_epoch=val_check_interval
    )

