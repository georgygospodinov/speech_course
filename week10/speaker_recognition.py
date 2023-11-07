from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple, Dict

import torch
import torchaudio
from tqdm.auto import tqdm


class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder: Path):
        super().__init__()

        self.wav_files = list(dataset_folder.rglob("*.wav"))
        self.labels = []
        self.cached_data: Dict[int, Tuple] = {}

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
        if idx not in self.cached_data:
            wav, _ = torchaudio.load(self.wav_files[idx])
            self.cached_data[idx] = (idx, wav[0], self.labels[idx])

        return self.cached_data[idx]


class StackingSubsampling(torch.nn.Module):
    def __init__(self, stride, feat_in, feat_out):
        super().__init__()
        self.stride = stride
        self.out = torch.nn.Linear(stride * feat_in, feat_out)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, wavs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

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
    dataset_folder: Path
    checkpoints_folder: Path
    model_params: ModelParams
    device: str = "cuda"
    n_epochs: int = 100
    batch_size: int = 16
    num_workers: int = 3
    learning_rate: float = 1e-2


def main(conf: ModuleParams) -> None:

    dataset = SpeakerDataset(dataset_folder=conf.dataset_folder)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=True
    )

    n_classes = len(set(dataset.labels))

    model_params = conf.model_params
    model_params = replace(model_params, n_classes=n_classes)

    model = Conformer(model_params).to(conf.device)

    optim = torch.optim.Adam(params=model.parameters(), lr=conf.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    pbar = tqdm(range(conf.n_epochs), position=0, leave=True)

    for epoch in pbar:

        for batch in train_dataloader:
            _, wavs, labels = batch

            _, scores = model.forward(wavs.to(conf.device))

            optim.zero_grad()
            loss = criterion(input=scores, target=labels.to(conf.device))
            loss.backward()
            optim.step()

            pbar.set_postfix({"batch_loss": f"{loss.item():.2f}"})

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": loss.item(),
            },
            conf.checkpoints_folder / f"epoch_{epoch + 1}.ckpt",
        )


if __name__ == "__main__":

    params = ModuleParams(
        dataset_folder=Path("./data/train"),
        checkpoints_folder=Path("./checkpoints"),
        model_params=ModelParams(),
        device="cpu",
    )
    main(params)
