import logging
from typing import Any

import torch
from torch import nn
import pytorch_lightning as pl

from src.encoder import Conformer
from src.decoder import ConvDecoder
from src.metrics import WER
from src.data import ASRDataset, collate_fn
from src.optim import get_scheduler, get_optimizer


logger = logging.getLogger("lightning")


class ConformerCTC(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf

        self.encoder = Conformer(conf.model.encoder)
        self.decoder = ConvDecoder(conf.model.decoder)

        self.log_every_n_steps = (
            conf.trainer.log_every_n_steps if "log_every_n_steps" in conf.trainer else 1
        )

        self.wer = WER()

        self.ctc_loss = nn.CTCLoss(
            blank=self.decoder.blank_id, zero_infinity=True, reduction="none"
        )

    def forward(
        self, features: torch.Tensor, features_len: torch.Tensor
    ) -> torch.Tensor:

        encoded, encoded_len = self.encoder(features, features_len)

        encoded = encoded.transpose(1, 2)  # B x T x D -> B x D x T

        logprobs = self.decoder(encoded)
        preds = logprobs.argmax(dim=-1)

        return logprobs, encoded_len, preds

    def training_step(self, batch: Any, batch_nb: int):
        features, features_len, targets, target_len = batch

        logprobs, encoded_len, preds = self.forward(features, features_len)

        loss = self.ctc_loss(
            logprobs.transpose(1, 0), targets, encoded_len, target_len
        ).mean()

        log = {"train_loss": loss, "lr": self.optimizers().param_groups[0]["lr"]}

        if (batch_nb + 1) % self.log_every_n_steps == 0:

            refs = self.decoder.decode(token_ids=targets, token_ids_length=target_len)
            hyps = self.decoder.decode(
                token_ids=preds, token_ids_length=encoded_len, unique_consecutive=True
            )
            logger.info("reference : %s", refs[0])
            logger.info("prediction: %s", hyps[0])
            self.wer.update(references=refs, hypotheses=hyps)
            wer, _, _ = self.wer.compute()

            self.wer.reset()

            log["train_wer"] = wer

        self.log_dict(log)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_nb):
        features, features_len, targets, target_len = batch

        logprobs, encoded_len, preds = self.forward(features, features_len)

        loss = self.ctc_loss(logprobs.transpose(1, 0), targets, encoded_len, target_len)

        refs = self.decoder.decode(token_ids=targets, token_ids_length=target_len)
        hyps = self.decoder.decode(
            token_ids=preds, token_ids_length=encoded_len, unique_consecutive=True
        )
        logger.info("reference : %s", refs[0])
        logger.info("prediction: %s", hyps[0])
        self.wer.update(references=refs, hypotheses=hyps)
        _, word_errors, words = self.wer.compute()

        self.wer.reset()

        return {"val_loss": loss, "val_word_errors": word_errors, "val_words": words}

    def validation_epoch_end(self, outputs):
        word_errors = torch.stack([x["val_word_errors"] for x in outputs]).sum()
        words = torch.stack([x["val_words"] for x in outputs]).sum()
        val_loss = torch.cat([x["val_loss"] for x in outputs]).mean()
        self.log_dict(
            {"val_wer": word_errors / words if words > 0 else -1, "val_loss": val_loss}
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            ASRDataset(self.conf.train_dataloader.dataset),
            batch_size=self.conf.train_dataloader.batch_size,
            num_workers=self.conf.train_dataloader.num_workers,
            prefetch_factor=self.conf.train_dataloader.prefetch_factor,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            ASRDataset(self.conf.val_dataloader.dataset),
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
