import logging
from typing import Any

import torch
from torch import nn
import pytorch_lightning as pl

from src.encoder.conformer import Conformer
from src.decoder.las import TransformerDecoder
from src.metrics import WER
from src.data import ASRDatasetBPE, collate_fn
from src.optim import get_scheduler, get_optimizer


logger = logging.getLogger("lightning")


class ConformerLAS(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf

        self.encoder = Conformer(conf.model.encoder)
        self.decoder = TransformerDecoder(conf.model.decoder)

        self.log_every_n_steps = (
            conf.trainer.log_every_n_steps if "log_every_n_steps" in conf.trainer else 1
        )

        self.wer = WER()

        self.loss = nn.CrossEntropyLoss(reduction="none")

    @staticmethod
    def make_attention_mask(length: torch.Tensor) -> torch.Tensor:
        max_len = length.max()
        return torch.triu(torch.ones((max_len, max_len)), diagonal=1) * (-10000.0)

    @staticmethod
    def make_pad_mask(length: torch.Tensor) -> torch.Tensor:
        return torch.arange(0, length.max()) < length.unsqueeze(1)

    def forward(
        self, features: torch.Tensor, features_len: torch.Tensor
    ) -> torch.Tensor:
        encoded, encoded_len = self.encoder(features, features_len)
        return encoded, encoded_len

    def training_step(self, batch: Any, batch_nb: int):
        features, features_len, targets, target_len = batch

        encoded, encoded_len = self.forward(features, features_len)
        encoded_pad_mask = self.make_pad_mask(encoded_len)

        targets_outputs = targets[:, 1:]  # without bos
        targets_inputs = targets[:, :-1]  # without eos / last pad token
        target_len -= 1

        target_pad_mask = self.make_pad_mask(target_len)
        target_mask = self.make_attention_mask(target_len)

        logits = self.decoder(
            encoded, ~encoded_pad_mask, targets_inputs, target_mask, ~target_pad_mask
        )

        loss = self.loss(logits.transpose(1, 2), targets_outputs)
        loss = (loss * target_pad_mask).sum() / target_pad_mask.sum()

        log = {"train_loss": loss, "lr": self.optimizers().param_groups[0]["lr"]}

        self.log_dict(log)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_nb):
        features, features_len, targets, target_len = batch

        encoded, encoded_len = self.forward(features, features_len)
        encoded_pad_mask = self.make_pad_mask(encoded_len)

        targets_outputs = targets[:, 1:]  # without bos
        targets_inputs = targets[:, :-1]  # without eos / last pad token
        target_len -= 1

        target_pad_mask = self.make_pad_mask(target_len)
        target_mask = self.make_attention_mask(target_len)

        logits = self.decoder(
            encoded, ~encoded_pad_mask, targets_inputs, target_mask, ~target_pad_mask
        )

        loss = self.loss(logits.transpose(1, 2), targets_outputs)
        loss = (loss * target_pad_mask).sum() / target_pad_mask.sum()
        return {"val_loss": loss}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            ASRDatasetBPE(self.conf.train_dataloader.dataset),
            batch_size=self.conf.train_dataloader.batch_size,
            num_workers=self.conf.train_dataloader.num_workers,
            prefetch_factor=self.conf.train_dataloader.prefetch_factor,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            ASRDatasetBPE(self.conf.val_dataloader.dataset),
            batch_size=self.conf.val_dataloader.batch_size,
            num_workers=self.conf.val_dataloader.num_workers,
            prefetch_factor=self.conf.train_dataloader.prefetch_factor,
            collate_fn=collate_fn,
        )

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.conf.optim.optimizer.name,
            params=self.parameters(),
            **self.conf.optim.optimizer.params,
        )
        lr_scheduler = {
            "scheduler": get_scheduler(
                self.conf.optim.scheduler.name,
                optimizer=optimizer,
                **self.conf.optim.scheduler.params,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
