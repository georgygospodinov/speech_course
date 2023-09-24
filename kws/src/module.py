from typing import Tuple

import hydra
import pytorch_lightning as pl
import torch
import thop
from torchmetrics import Accuracy


class KWS(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()

        self.conf = conf

        self.model = hydra.utils.instantiate(conf.model)
        self.train_acc = Accuracy(
            task="multiclass", num_classes=conf.model.n_classes, top_k=1
        )
        self.valid_acc = Accuracy(
            task="multiclass", num_classes=conf.model.n_classes, top_k=1
        )

        self.loss = hydra.utils.instantiate(conf.loss)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(inputs)
        preds = logits.argmax(1)
        return logits, preds

    def on_train_start(self):

        features_params = self.conf.train_dataloader.dataset.transforms[0]

        sample_inputs = torch.randn(
            1,
            features_params.n_mels,
            features_params.sample_rate // features_params.hop_length + 1,
            device=self.device,
        )
        macs, params = thop.profile(
            self.model,
            inputs=(sample_inputs,),
        )
        self.log("MACs", macs)
        self.log("Params", params)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        logits, preds = self.forward(inputs)

        loss = self.loss(logits, labels)

        log = {
            "train/loss": loss,
            "lr": self.optimizers().param_groups[0]["lr"],
            "train/accuracy": self.train_acc(preds, labels),
        }

        self.log_dict(log, on_step=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        logprobs, preds = self.forward(inputs)

        loss = self.loss(logprobs, labels)
        self.valid_acc.update(preds, labels)

        return {"loss": loss}

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        ids, inputs, _ = batch

        _, preds = self.forward(inputs)

        return ids, preds

    def on_validation_epoch_end(self):
        self.log("val/accuracy", self.valid_acc.compute())
        self.valid_acc.reset()

    def train_dataloader(self):
        return hydra.utils.instantiate(self.conf.train_dataloader)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.conf.val_dataloader)

    def predict_dataloader(self):
        return hydra.utils.instantiate(self.conf.predict_dataloader)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.conf.optim,
            params=self.model.parameters(),
        )
        return {"optimizer": optimizer}


class KDKWS(KWS):
    def __init__(self, conf, teacher_weights):
        super().__init__(conf)

        self.teacher_model = hydra.utils.instantiate(conf.teacher_module.model)

        ckpt = torch.load(teacher_weights, map_location="cpu")
        self.teacher_model.load_state_dict({
            k[len("model."):]: v for k,v in ckpt["state_dict"].items()
            if "total" not in k
        })
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

        self.kd_loss = hydra.utils.instantiate(conf.teacher_module.kd_loss)
        self.alpha = conf.teacher_module.kd_weight

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        student_logits, preds = self.forward(inputs)
        teacher_logits = self.teacher_model(inputs)

        student_loss = self.loss(student_logits, labels)
        kd_loss = self.kd_loss(student_logits, teacher_logits)

        loss = (1 - self.alpha) * student_loss + self.alpha * kd_loss

        log = {
            "train/loss": student_loss,
            "train/total_loss": loss,
            "train/kd_loss": kd_loss,
            "lr": self.optimizers().param_groups[0]["lr"],
            "train/accuracy": self.train_acc(preds, labels),
        }

        self.log_dict(log, on_step=True)

        return {"loss": loss}
