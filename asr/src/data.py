from pathlib import Path

import torch
import torchaudio
import sentencepiece
import pandas as pd

from src.transforms import get_transform


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, conf):
        super().__init__()

        self.transform = get_transform(conf.transforms)

        manifest_path = Path(__file__).parent.parent / "data" / conf.manifest_name

        manifest = pd.read_json(manifest_path, lines=True)

        self.wav_files = [
            manifest_path.parent / wav_path for wav_path in manifest.audio_filepath
        ]

        token_to_idx = {token: idx for idx, token in enumerate(conf.labels)}

        self.targets = [
            [token_to_idx[token] for token in text] for text in manifest.text
        ]

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav, _ = torchaudio.load(self.wav_files[idx])
        features = self.transform(wav)[0]
        target = self.targets[idx]
        return features.T, features.shape[1], torch.Tensor(target), len(target)


class ASRDatasetBPE(torch.utils.data.Dataset):
    def __init__(self, conf):
        super().__init__()

        self.transform = get_transform(conf.transforms)
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=conf.tokenizer)

        manifest_path = Path(__file__).parent.parent / "data" / conf.manifest_name

        manifest = pd.read_json(manifest_path, lines=True)
        if "max_duration" in conf:
            manifest = manifest[manifest.duration <= conf.max_duration].reset_index(
                drop=True
            )

        if "min_duration" in conf:
            manifest = manifest[manifest.duration >= conf.min_duration].reset_index(
                drop=True
            )

        if "max_len" in conf:
            manifest = manifest[manifest.text.str.len() <= conf.max_len].reset_index(
                drop=True
            )

        self.wav_files = [
            manifest_path.parent / wav_path for wav_path in manifest.audio_filepath
        ]
        self.texts = manifest.text.values

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav, _ = torchaudio.load(self.wav_files[idx])
        features = self.transform(wav)[0]
        target = (
            [self.tokenizer.bos_id()]
            + self.tokenizer.encode(self.texts[idx])
            + [self.tokenizer.eos_id()]
        )
        return features.T, features.shape[1], torch.Tensor(target), len(target)


def collate_fn(batch):
    features, features_length, targets, targets_length = list(zip(*batch))
    features_padded = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True
    ).permute(0, 2, 1)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return (
        features_padded,
        torch.Tensor(features_length).long(),
        targets_padded.long(),
        torch.Tensor(targets_length).long(),
    )
