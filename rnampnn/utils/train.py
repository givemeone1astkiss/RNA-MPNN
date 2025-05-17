import numpy as np
import pytorch_lightning as pl
from ..config.glob import OUTPUT_PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback
import torch
from ..model.rnampnn import RNAMPNN
from tqdm import tqdm
import pickle

class LossMonitor(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, model: RNAMPNN):
        avg_loss = torch.tensor([x['val_loss'] * x['len'] for x in model.val_step_outputs]).sum(dim=-1).to(device=model.device) / torch.tensor(
            [x['len'] for x in model.val_step_outputs]).sum(dim=-1).to(device=model.device)
        avg_recovery_rate = torch.tensor([x['correct'] for x in model.val_step_outputs]).sum(dim=-1).to(device=model.device) / torch.tensor(
            [x['len'] for x in model.val_step_outputs]).sum(dim=-1).to(device=model.device)

        model.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        model.log('val_recovery_rate', avg_recovery_rate, prog_bar=True, sync_dist=True)
        model.val_step_outputs = []

    def on_test_epoch_end(self, trainer: pl.Trainer, model: RNAMPNN):
        avg_loss = torch.tensor([x['test_loss'] * x['len'] for x in model.test_step_outputs]).sum(dim=-1).to(device=model.device) / torch.tensor(
            [x['len'] for x in model.test_step_outputs]).sum(dim=-1).to(device=model.device)
        avg_recovery_rate = torch.tensor([x['correct'] for x in model.test_step_outputs]).sum(dim=-1).to(device=model.device) / torch.tensor(
            [x['len'] for x in model.test_step_outputs]).sum(dim=-1).to(device=model.device)

        model.log('test_loss', avg_loss, prog_bar=True, sync_dist=True)
        model.log('test_recovery_rate', avg_recovery_rate, prog_bar=True, sync_dist=True)
        model.test_step_outputs = []

class NameModel(Callback):
    def __init__(self, name: str, version: int):
        super().__init__()
        self.name = name
        self.version = version

    def on_fit_start(self, trainer: pl.Trainer, model: RNAMPNN) -> None:
        model.name = self.name
        model.version = self.version

class XGBTrainer(Callback):
    def __init__(self):
        super().__init__()
        self.batch_val_loss = []
        self.batch_val_length = []
        self.batch_val_correct = []

    def on_fit_end(self, trainer: pl.Trainer, model: RNAMPNN) -> None:
        print('Start training XGBoost!\n')
        X, Y = self._generate_embedding(trainer, model)
        model.xgb_readout.fit(X, Y)
        for val_batch in tqdm(trainer.val_dataloaders, desc="Validating XGBoost", total=len(trainer.val_dataloaders), position=1):
            self._xgb_valid_step(model, val_batch)
        train_loss = model.xgb_readout.score(X, Y)
        print(f'Training score: {train_loss}\n')
        val_loss = ((torch.tensor(self.batch_val_length) * torch.tensor(self.batch_val_loss)).sum(dim=-1) / torch.tensor(self.batch_val_length).sum(dim=-1)).to(device=model.device)
        val_recovery_rate = torch.tensor(self.batch_val_correct).sum(dim=-1) / torch.tensor(self.batch_val_length).sum(dim=-1)
        print(f'Validation score: {val_loss.item()}\n')
        print(f'Validation recovery rate: {val_recovery_rate.item()}\n')
        self.batch_val_loss = []
        self.batch_val_length = []
        self.batch_val_correct = []
        print('XGBoost training done!')

        with open(f"{OUTPUT_PATH}/checkpoints/{model.name}/XGB-V{model.version}.pkl", 'wb') as f:
            pickle.dump(model.xgb_readout, f)

    @staticmethod
    def _generate_embedding( trainer: pl.Trainer, model: RNAMPNN):
        X = np.ndarray((0, model.hparams.res_embedding_dim))
        Y = np.ndarray((0))
        for batch in tqdm(trainer.train_dataloader, desc="Generating Embedding...", total=len(trainer.train_dataloader), position=0):
            sequences, coords, mask, _ = batch
            sequences = sequences.to(device=model.device)
            coords = coords.to(device=model.device)
            mask = mask.to(device=model.device)
            embedding = model.embedding(coords, mask)[mask.bool()]
            sequences = torch.argmax(sequences[mask.bool()], dim=-1)
            X = np.append(X, embedding.to(device=torch.device(torch.device('cpu'))).detach().numpy(), axis=0)
            Y = np.append(Y, sequences.to(device=torch.device(torch.device('cpu'))).detach().numpy(), axis=0)
        return X, Y

    def _xgb_valid_step(self, model: RNAMPNN, batch):
        sequences, coords, mask, _ = batch
        sequences = sequences.to(device=model.device)
        coords = coords.to(device=model.device)
        mask = mask.to(device=model.device)
        embedding = model.embedding(coords, mask)[mask.bool()].to(device=torch.device(torch.device('cpu')))
        sequences = torch.argmax(sequences[mask.bool()], dim=-1).to(device=torch.device(torch.device('cpu')))
        self.batch_val_loss.append(model.xgb_readout.score(embedding.detach().numpy(), sequences.detach().numpy()))
        self.batch_val_correct.append(np.equal(model.xgb_readout.predict(embedding.detach().numpy()),sequences.detach().numpy()).sum())
        self.batch_val_length.append(sequences.shape[0])

def get_trainer(name: str, version: int, max_epochs: int=60, val_check_interval: int = 1, progress_bar=True):
    logger = pl.loggers.TensorBoardLogger(
        save_dir=f"{OUTPUT_PATH}/logs",
        name=name,
        version=version,
    )

    checkpoint = ModelCheckpoint(
        dirpath=f"{OUTPUT_PATH}/checkpoints/{name}/",
        filename='{epoch:02d}'+f'-{version}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        verbose=True
    )

    return pl.Trainer(
        accelerator="auto",
        devices=-1 if torch.cuda.is_available() else 1,
        precision="bf16-mixed",
        max_epochs=max_epochs,
        enable_progress_bar=progress_bar,
        logger=logger,
        callbacks=[checkpoint, LossMonitor(), XGBTrainer(), NameModel(name, version)],
        check_val_every_n_epoch=val_check_interval,
        strategy='ddp_find_unused_parameters_true',
        enable_checkpointing=True
    )

                