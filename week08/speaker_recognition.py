from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path, use_cache: bool = True):
        super().__init__()

        self.wav_files = list(dataset_dir.rglob("*.wav"))
        self.labels = []
        self.cached_data: dict[int, tuple] = {}
        self.use_cache = use_cache

        class2idx = {}
        last_class_idx = -1
        for path in self.wav_files:
            class_name = path.parent.stem

            if class_name not in class2idx:
                last_class_idx += 1
                class2idx[class_name] = last_class_idx
            self.labels.append(class2idx[class_name])

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        if self.use_cache:
            if idx not in self.cached_data:
                wav, _ = torchaudio.load(self.wav_files[idx])
                self.cached_data[idx] = (idx, wav[0], self.labels[idx])
            return self.cached_data[idx]
        else:
            wav, _ = torchaudio.load(self.wav_files[idx])
            return (idx, wav[0], self.labels[idx])


class StackingSubsampling(torch.nn.Module):
    def __init__(self, stride, feat_in, feat_out):
        super().__init__()
        self.stride = stride
        self.out = torch.nn.Linear(stride * feat_in, feat_out)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = features.size()
        pad_size = (self.stride - (t % self.stride)) % self.stride
        features = torch.nn.functional.pad(features, (0, 0, 0, pad_size))
        _, t, _ = features.size()
        features = torch.reshape(features, (b, t // self.stride, d * self.stride))
        out_features = self.out(features)
        out_length = torch.div(
            features_length + pad_size, self.stride, rounding_mode="floor"
        )
        return out_features, out_length


class StatisticsPooling(torch.nn.Module):
    @staticmethod
    def get_length_mask(length):
        """
        length: B
        """
        max_len = length.max().long().item()

        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        return mask.to(length.dtype)

    def forward(self, encoded, encoded_len):
        """
        encoded: B x T x D
        encoded_len: B
        return: B x 2D
        """

        mask = self.get_length_mask(encoded_len).unsqueeze(2)  # B x T x 1

        total = encoded_len.unsqueeze(1)

        avg = (encoded * mask).sum(dim=1) / total

        std = torch.sqrt(
            (mask * (encoded - avg.unsqueeze(dim=1)) ** 2).sum(dim=1) / total
        )

        return torch.cat((avg, std), dim=1)


class AngularMarginSoftmax(torch.nn.Module):
    """
    Angular Margin Softmax Loss
    https://arxiv.org/abs/1906.07317
    """

    def __init__(
        self, embedding_dim: int, num_classes: int, margin: float, scale: float
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: B x D
        labels: B
        return: scalar tensor
        """
        raise NotImplementedError

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: B x D
        return: B
        """
        raise NotImplementedError


class SpecScaler(torch.nn.Module):
    def forward(self, spectrogram):
        return torch.log(spectrogram.clamp_(1e-9, 1e9))


class Conformer(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=conf.sample_rate,
                n_fft=conf.n_fft,
                win_length=conf.win_length,
                hop_length=conf.hop_length,
                n_mels=conf.n_mels,
            ),
            SpecScaler(),
        )
        self.subsampling = StackingSubsampling(conf.stride, conf.feat_in, conf.d_model)
        self.backbone = torchaudio.models.Conformer(
            input_dim=conf.d_model,
            num_heads=conf.n_heads,
            ffn_dim=conf.d_model * conf.ff_exp_factor,
            num_layers=conf.n_layers,
            depthwise_conv_kernel_size=conf.kernel_size,
            dropout=conf.dropout,
        )
        self.pooler = StatisticsPooling()
        self.extractor = torch.nn.Sequential(
            torch.nn.Linear(2 * conf.d_model, conf.d_model),
            torch.nn.ELU(),
            torch.nn.Linear(conf.d_model, conf.emb_size),
            torch.nn.ELU(),
        )
        self.proj = torch.nn.Sequential(torch.nn.Linear(conf.emb_size, conf.n_classes))

    def forward(self, wavs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.transform(wavs)

        features_length = (
            torch.ones(features.shape[0], device=features.device) * features.shape[2]
        ).to(torch.long)

        features = features.transpose(1, 2)  # B x D x T -> B x T x D
        features, features_length = self.subsampling(features, features_length)
        encoded, encoded_len = self.backbone(features, features_length)
        emb = self.pooler(encoded, encoded_len)
        emb = self.extractor(emb)
        scores = self.proj(emb)
        return emb, scores


@dataclass
class ModelParams:
    stride: int = 8
    feat_in: int = 64
    d_model: int = 32
    n_heads: int = 4
    ff_exp_factor: int = 2
    n_layers: int = 2
    kernel_size: int = 5
    dropout: float = 0.0
    emb_size: int = 16
    n_classes: int = 377
    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64


@dataclass
class ModuleParams:
    dataset_dir: Path
    checkpoints_dir: Path
    log_dir: Path
    model_params: ModelParams
    angular_margin: float | None = None
    angular_scale: float | None = None
    use_cache: bool = True
    device: str = "cuda"
    n_epochs: int = 100
    batch_size: int = 16
    num_workers: int = 3
    learning_rate: float = 1e-2
    loss_function: str = "cross_entropy"  # "cross_entropy" or "angular_margin"
    validation_dir: Path | None = None
    validation_frequency: int = 5


def evaluate(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> float:
    raise NotImplementedError


def main(conf: ModuleParams) -> None:

    conf.log_dir.mkdir(exist_ok=True, parents=True)
    conf.checkpoints_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=conf.log_dir)

    dataset = SpeakerDataset(dataset_dir=conf.dataset_dir, use_cache=conf.use_cache)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=True
    )

    val_dataloader = None
    if conf.validation_dir and conf.validation_dir.exists():
        val_dataset = SpeakerDataset(
            dataset_dir=conf.validation_dir, use_cache=conf.use_cache
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=conf.batch_size,
            num_workers=conf.num_workers,
            shuffle=False,
        )

    n_classes = len(set(dataset.labels))

    model_params = conf.model_params
    model_params = replace(model_params, n_classes=n_classes)

    model = Conformer(model_params).to(conf.device)

    if conf.loss_function == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif conf.loss_function == "angular_margin":
        criterion = AngularMarginSoftmax(
            embedding_dim=model_params.emb_size,
            num_classes=n_classes,
            margin=conf.angular_margin,
            scale=conf.angular_scale,
        ).to(conf.device)
    else:
        raise ValueError(f"Invalid loss function: {conf.loss_function}")

    optim = torch.optim.Adam(params=model.parameters(), lr=conf.learning_rate)

    pbar = tqdm(range(conf.n_epochs), position=0, leave=True)

    global_step = 0

    for epoch in pbar:
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch in train_dataloader:
            _, wavs, labels = batch

            _, scores = model.forward(wavs.to(conf.device))

            optim.zero_grad()

            loss = criterion(scores, labels.to(conf.device))

            loss.backward()
            optim.step()

            predictions = torch.argmax(scores, dim=1)

            correct = (predictions == labels.to(conf.device)).sum().item()
            epoch_correct += correct
            epoch_total += labels.size(0)

            epoch_losses.append(loss.item())

            writer.add_scalar("Loss/Batch", loss.item(), global_step)
            writer.add_scalar("Accuracy/Batch", correct / labels.size(0), global_step)
            writer.add_scalar("Learning_Rate", optim.param_groups[0]["lr"], global_step)

            global_step += 1

            pbar.set_postfix({"batch_loss": f"{loss.item():.2f}"})

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_accuracy = epoch_correct / epoch_total

        writer.add_scalar("Loss/Epoch", avg_epoch_loss, epoch)
        writer.add_scalar("Accuracy/Epoch", epoch_accuracy, epoch)

        if val_dataloader and (epoch + 1) % conf.validation_frequency == 0:
            print(f"\nRunning validation evaluation at epoch {epoch + 1}...")
            try:
                eer = evaluate(model, val_dataloader, conf.device)
            except NotImplementedError:
                eer = -1
            writer.add_scalar("Validation/EER", eer, epoch)

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": avg_epoch_loss,
                "accuracy": epoch_accuracy,
                "loss_function": conf.loss_function,
                "angular_margin": (
                    conf.angular_margin
                    if conf.loss_function == "angular_margin"
                    else None
                ),
                "angular_scale": (
                    conf.angular_scale
                    if conf.loss_function == "angular_margin"
                    else None
                ),
            },
            conf.checkpoints_dir / f"epoch_{epoch + 1}.ckpt",
        )

    writer.close()


if __name__ == "__main__":

    params = ModuleParams(
        dataset_dir=Path("./data/train"),
        use_cache=False,
        checkpoints_dir=Path("./checkpoints"),
        model_params=ModelParams(),
        device="cpu",
        num_workers=1,
        n_epochs=20,
        log_dir=Path("./logs/cross_entropy_hw_test"),
        loss_function="cross_entropy",
        validation_dir=Path("./data/dev"),
        validation_frequency=1,
    )
    main(params)
