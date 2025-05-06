import pytorch_lightning as pl
from ..config.glob import OUTPUT_PATH
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

def get_trainer(name: str, version: int, max_epochs: int=60, val_check_interval: int = 1):
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
        enable_progress_bar=True,
        logger=logger,
        callbacks=[checkpoint],
        check_val_every_n_epoch=val_check_interval
    )