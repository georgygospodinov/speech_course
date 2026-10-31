import argparse
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import roc_curve
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path, use_cache: bool = True):
        super().__init__()

        self.wav_files = sorted(dataset_dir.rglob("*.wav"))
        if not self.wav_files:
            raise FileNotFoundError(f"No WAV files found in {dataset_dir.resolve()}")
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
        if idx in self.cached_data:
            return self.cached_data[idx]
        wav, sample_rate = torchaudio.load(self.wav_files[idx], backend="soundfile")
        wav = wav.mean(dim=0)
        if sample_rate != 16_000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16_000)
        if not wav.numel():
            raise ValueError(f"Empty recording: {self.wav_files[idx]}")
        item = (idx, wav, self.labels[idx])
        if self.use_cache:
            self.cached_data[idx] = item
        return item


def collate_speakers(batch):
    """Pad waveforms while keeping real lengths for the encoder and pooling."""
    indices, wavs, labels = zip(*batch)
    lengths = torch.tensor([wav.numel() for wav in wavs])
    return (
        torch.tensor(indices),
        torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True),
        torch.tensor(labels),
        lengths,
    )


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
            features_length + self.stride - 1, self.stride, rounding_mode="floor"
        )
        return out_features, out_length


class StatisticsPooling(torch.nn.Module):
    @staticmethod
    def get_length_mask(length, max_len=None):
        """
        length: B
        """
        if max_len is None:
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

        mask = self.get_length_mask(encoded_len, encoded.size(1)).unsqueeze(2)

        total = encoded_len.unsqueeze(1)

        avg = (encoded * mask).sum(dim=1) / total

        std = torch.sqrt(
            ((mask * (encoded - avg.unsqueeze(dim=1)) ** 2).sum(dim=1) / total)
            .clamp_min(1e-9)
        )

        return torch.cat((avg, std), dim=1)


class AngularMarginSoftmax(torch.nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace): s * cos(theta_y + margin).
    https://arxiv.org/abs/1906.07317
    """

    def __init__(
        self, embedding_dim: int, num_classes: int, margin: float, scale: float
    ):
        super().__init__()
        if margin is None or not 0 <= margin < math.pi / 2:
            raise ValueError("margin must be in [0, pi/2), in radians")
        if scale is None or scale <= 0:
            raise ValueError("scale must be positive")
        self.weight = torch.nn.Parameter(torch.empty(num_classes, embedding_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def cosine(self, embeddings):
        return F.linear(F.normalize(embeddings), F.normalize(self.weight))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: B x D
        labels: B
        return: scalar tensor
        """
        cosine = self.cosine(embeddings).clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt((1 - cosine.square()).clamp_min(1e-7))
        target_cosine = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        # Keep the target logit monotone past pi - margin.
        target_cosine = torch.where(
            cosine > math.cos(math.pi - self.margin),
            target_cosine,
            cosine - math.sin(self.margin) * self.margin,
        )
        target_mask = F.one_hot(labels, num_classes=self.weight.size(0)).bool()
        logits = torch.where(target_mask, target_cosine, cosine) * self.scale
        return F.cross_entropy(logits, labels)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: B x D
        return: B
        """
        return self.cosine(embeddings).argmax(dim=1)


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

    def forward(
        self, wavs: torch.Tensor, wav_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.transform(wavs)

        if wav_lengths is None:
            features_length = torch.full(
                (features.shape[0],), features.shape[2], device=features.device,
                dtype=torch.long,
            )
        else:
            # MelSpectrogram uses center=True.
            features_length = wav_lengths.to(features.device) // self.transform[0].hop_length + 1

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


def verification_pairs(embeddings, labels):
    """Each unordered pair once; exclude self comparisons (the diagonal)."""
    embeddings = F.normalize(torch.as_tensor(embeddings).float(), dim=1)
    labels = np.asarray(labels)
    rows, cols = np.triu_indices(len(labels), k=1)
    scores = (embeddings @ embeddings.T).cpu().numpy()
    return rows, cols, scores[rows, cols], labels[rows] == labels[cols]


def verification_curve(pair_scores, same_speaker):
    """Empirical FAR/FRR for score >= threshold, with ties handled together.

    EER is the mean FAR/FRR at the closest empirical operating point.
    Values are fractions (0.05 means 5%); the threshold is dev-calibrated.
    """
    if len(np.unique(same_speaker)) != 2:
        raise ValueError("Need both same-speaker and different-speaker pairs")
    far, true_accept_rate, thresholds = roc_curve(
        same_speaker, pair_scores, drop_intermediate=False
    )
    frr = 1 - true_accept_rate
    idx = np.abs(far - frr).argmin()
    return {
        "far": far, "frr": frr, "thresholds": thresholds,
        "eer": float((far[idx] + frr[idx]) / 2),
        "threshold": float(thresholds[idx]), "index": int(idx),
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> float:
    was_training = model.training
    model.eval()
    embeddings, all_labels = [], []
    try:
        for batch in dataloader:
            _, wavs, labels, *lengths = batch
            wav_lengths = lengths[0].to(device) if lengths else None
            emb, _ = model(wavs.to(device), wav_lengths)
            embeddings.append(emb.cpu())
            all_labels.extend(labels.tolist())
        _, _, pair_scores, same = verification_pairs(torch.cat(embeddings), all_labels)
        return verification_curve(pair_scores, same)["eer"]
    finally:
        model.train(was_training)


def main(conf: ModuleParams) -> None:

    conf.log_dir.mkdir(exist_ok=True, parents=True)
    conf.checkpoints_dir.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(log_dir=conf.log_dir)

    dataset = SpeakerDataset(dataset_dir=conf.dataset_dir, use_cache=conf.use_cache)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=True,
        collate_fn=collate_speakers,
        generator=torch.Generator().manual_seed(42),
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
            collate_fn=collate_speakers,
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

    optim = torch.optim.Adam(
        params=list(model.parameters()) + list(criterion.parameters()),
        lr=conf.learning_rate,
    )

    pbar = tqdm(range(conf.n_epochs), position=0, leave=True)

    global_step = 0
    best_eer = float("inf")

    for epoch in pbar:
        epoch_start = time.perf_counter()
        model.train()
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch in train_dataloader:
            _, wavs, labels, wav_lengths = batch
            labels = labels.to(conf.device)

            embeddings, scores = model(wavs.to(conf.device), wav_lengths.to(conf.device))

            optim.zero_grad()

            if conf.loss_function == "angular_margin":
                loss = criterion(embeddings, labels)
                predictions = criterion.predict(embeddings)
            else:
                loss = criterion(scores, labels)
                predictions = scores.argmax(dim=1)

            loss.backward()
            optim.step()

            correct = (predictions == labels).sum().item()
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
        writer.add_scalar("Time/Epoch_seconds", time.perf_counter() - epoch_start, epoch)
        print(f"Epoch {epoch + 1}: loss={avg_epoch_loss:.4f}, accuracy={epoch_accuracy:.2%}", flush=True)

        eer = None
        if val_dataloader and (epoch + 1) % conf.validation_frequency == 0:
            print(f"\nRunning validation evaluation at epoch {epoch + 1}...")
            eer = evaluate(model, val_dataloader, conf.device)
            writer.add_scalar("Validation/EER", eer, epoch)
            print(f"Dev EER: {eer:.2%}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "model_params": asdict(model_params),
            "criterion_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "global_step": global_step,
            "class_names": list(dict.fromkeys(p.parent.stem for p in dataset.wav_files)),
            "loss": avg_epoch_loss,
            "accuracy": epoch_accuracy,
            "dev_eer": eer,
            "loss_function": conf.loss_function,
            "angular_margin": (
                conf.angular_margin if conf.loss_function == "angular_margin" else None
            ),
            "angular_scale": (
                conf.angular_scale if conf.loss_function == "angular_margin" else None
            ),
        }
        torch.save(checkpoint, conf.checkpoints_dir / f"epoch_{epoch + 1}.ckpt")
        if eer is not None and eer < best_eer:
            best_eer = eer
            torch.save(checkpoint, conf.checkpoints_dir / "best.ckpt")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional speaker encoder training")
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--dev-dir", type=Path, default=Path("data/dev"))
    parser.add_argument("--loss", choices=["cross_entropy", "angular_margin"], default="cross_entropy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    params = ModuleParams(
        dataset_dir=args.train_dir,
        use_cache=False,
        checkpoints_dir=args.checkpoints_dir or Path("checkpoints") / args.loss,
        model_params=ModelParams(),
        device=args.device,
        num_workers=0,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        angular_margin=args.margin,
        angular_scale=args.scale,
        log_dir=args.log_dir or Path("logs") / args.loss,
        loss_function=args.loss,
        validation_dir=args.dev_dir,
        validation_frequency=1,
    )
    main(params)
