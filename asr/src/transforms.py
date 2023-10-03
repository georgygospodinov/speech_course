from typing import Callable

import torch
import torchaudio


class SpecScaler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


def get_transform(conf) -> Callable:
    name_to_transform = {
        "mel_spectrogram": torchaudio.transforms.MelSpectrogram,
        "log_scaler": SpecScaler,
    }

    transform_list = []

    for transform in conf:
        callable_transform = name_to_transform[transform.name]

        transform_list.append(
            callable_transform(**transform.params)
            if "params" in transform
            else callable_transform()
        )

    return torch.nn.Sequential(*transform_list)
