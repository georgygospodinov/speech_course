from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torchaudio


class SpecScaler(torch.nn.Module):
    def forward(self, spectrogram):
        return torch.log(spectrogram.clamp_(1e-9, 1e9))


class SpotterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str,
        idx_to_keyword: List[str],
        transforms: List[torch.nn.Module],
        test: Optional[bool] = False,
    ):
        super().__init__()

        self.transform = torch.nn.Sequential(*transforms)

        manifest = pd.read_csv(manifest_path)
        parent = Path(manifest_path).parent

        self.wav_files = [parent / wav_path for wav_path in manifest.path]

        keyword_to_idx = {keyword: idx for idx, keyword in enumerate(idx_to_keyword)}
        self.labels = (
            [-1] * manifest.shape[0]
            if test
            else [keyword_to_idx[keyword] for keyword in manifest.label]
        )

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav, _ = torchaudio.load(self.wav_files[idx])
        features = self.transform(wav)
        return idx, wav[0], features, self.labels[idx]


def collator(
    data: List[Tuple[int, torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    ids, _, specs, labels = zip(*data)

    ids_tensor = torch.Tensor(ids).long()
    spec_tensor = torch.cat(specs)
    label_tensor = torch.Tensor(labels).long()

    return ids_tensor, spec_tensor, label_tensor
