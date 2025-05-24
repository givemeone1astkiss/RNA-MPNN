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
        avg_loss = torch.tensor(model.val_step_outputs['val_loss']).sum(dim=-1).to(device=model.device) / torch.tensor(
            model.val_step_outputs['len']).sum(dim=-1).to(device=model.device)
        weighted_recovery_rate = torch.tensor(model.val_step_outputs['correct']).sum(dim=-1).to(
            device=model.device) / torch.tensor(
            model.val_step_outputs['len']).sum(dim=-1).to(device=model.device)
        recovery_rate = (torch.tensor(model.val_step_outputs['recovery_rates'])).to(device=model.device).mean(dim=-1)

        model.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        model.log('weighted_val_recovery_rate', weighted_recovery_rate, prog_bar=True, sync_dist=True)
        model.log('val_recovery_rate', recovery_rate, prog_bar=True, sync_dist=True)
        model.val_step_outputs = {'val_loss':[], 'correct':[], 'len':[], 'recovery_rates':[]}

    def on_test_epoch_end(self, trainer: pl.Trainer, model: RNAMPNN):
        avg_loss = torch.tensor(model.test_step_outputs['test_loss']).sum(dim=-1).to(device=model.device) / torch.tensor(
            model.test_step_outputs['len']).sum(dim=-1).to(device=model.device)
        weighted_recovery_rate = torch.tensor(model.test_step_outputs['correct']).sum(dim=-1).to(
            device=model.device) / torch.tensor(
            model.test_step_outputs['len']).sum(dim=-1).to(device=model.device)
        recovery_rate = (torch.tensor(model.test_step_outputs['recovery_rates'])).to(device=model.device).mean(dim=-1)

        model.log('test_loss', avg_loss, prog_bar=True, sync_dist=True)
        model.log('weighted_test_recovery_rate', weighted_recovery_rate, prog_bar=True, sync_dist=True)
        model.log('test_recovery_rate', recovery_rate, prog_bar=True, sync_dist=True)
        model.test_step_outputs = {'test_loss':[], 'correct':[], 'len':[], 'recovery_rates':[]}


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
        model = RNAMPNN.load_from_checkpoint(f"{OUTPUT_PATH}checkpoints/{model.name}/Final-V{model.version}.ckpt")
        print('=' * 20, '\n')
        print('Start training XGBoost!\n')
        X, Y = self._generate_embedding(trainer.train_dataloader, model)
        model.xgb_readout.fit(X, Y)
        train_score = model.xgb_readout.score(X, Y)
        print(f'Training score: {train_score}\n')
        X, Y = self._generate_embedding(trainer.val_dataloaders, model)
        val_score = model.xgb_readout.score(X, Y)
        print('Start validation!\n')
        print(f'Validation score: {val_score}\n')
        print('XGBoost training done!')
        print('=' * 20, '\n')
        with open(f"{OUTPUT_PATH}checkpoints/{model.name}/XGB-V{model.version}.pkl", 'wb') as f:
            pickle.dump(model.xgb_readout, f)

    @staticmethod
    def _generate_embedding(dataloader: torch.utils.data.DataLoader, model: RNAMPNN):
        X = np.ndarray((0, model.hparams.res_embedding_dim+model.hparams.raw_embedding_dim))
        Y = np.ndarray(0)
        for batch in tqdm(dataloader, desc="Generating Embedding...", total=len(dataloader)):
            sequences, coords, mask, _ = batch
            sequences = sequences.to(device=model.device)
            coords = coords.to(device=model.device)
            mask = mask.to(device=model.device)
            embedding = model.embedding(coords, mask)[mask.bool()]
            sequences = torch.argmax(sequences[mask.bool()], dim=-1)
            X = np.append(X, embedding.to(device=torch.device(torch.device('cpu'))).detach().numpy(), axis=0)
            Y = np.append(Y, sequences.to(device=torch.device(torch.device('cpu'))).detach().numpy(), axis=0)
        return X, Y

def get_trainer(name: str, version: int, max_epochs: int=60, val_check_interval: int = 1, progress_bar=True):
    logger = pl.loggers.TensorBoardLogger(
        save_dir=f"{OUTPUT_PATH}/logs",
        name=name,
        version=version,
    )

    checkpoint = ModelCheckpoint(
        dirpath=f"{OUTPUT_PATH}checkpoints/{name}/",
        filename=f'Final-V{version}',
        save_top_k=1,
        monitor='val_recovery_rate',
        mode='max',
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

                