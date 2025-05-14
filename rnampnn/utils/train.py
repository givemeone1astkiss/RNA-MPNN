import pytorch_lightning as pl
from torch.xpu import device

from ..config.glob import OUTPUT_PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback
import torch
from ..model.rnampnn import RNAMPNN

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

def get_trainer(name: str, version: int, max_epochs: int=60, val_check_interval: int = 1, progress_bar=True):
    logger = pl.loggers.TensorBoardLogger(
        save_dir=f"{OUTPUT_PATH}/logs",
        name=name,
        version=version,
    )

    checkpoint = ModelCheckpoint(
        dirpath=f"{OUTPUT_PATH}/checkpoints/{name}/",
        filename='checkpoint-{epoch:02d}'+f'-{version}',
        save_top_k=1,
        verbose=True
    )

    return pl.Trainer(
        accelerator="auto",
        devices=-1 if torch.cuda.is_available() else 1,
        precision="bf16-mixed",
        max_epochs=max_epochs,
        enable_progress_bar=progress_bar,
        logger=logger,
        callbacks=[checkpoint, LossMonitor()],
        check_val_every_n_epoch=val_check_interval,
        strategy='ddp_find_unused_parameters_true',
        enable_checkpointing=True
    )

