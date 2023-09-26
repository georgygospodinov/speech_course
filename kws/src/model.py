from typing import List

import torch


class Conv1dNet(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        kernels: List[int],
        strides: List[int],
        channels: List[int],
        activation: torch.nn.Module,
        hidden_size: int,
    ):
        super().__init__()

        features = in_features

        module_list = []

        for kernel_size, stride, chs in zip(kernels, strides, channels):

            module_list.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=features,
                        out_channels=chs,
                        kernel_size=kernel_size,
                        stride=stride,
                        groups=chs,
                    ),
                    activation,
                    torch.nn.Conv1d(in_channels=chs, out_channels=chs, kernel_size=1),
                    torch.nn.BatchNorm1d(num_features=chs),
                    activation,
                    torch.nn.MaxPool1d(kernel_size=stride),
                ]
            )

            features = chs

        module_list.extend(
            [
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(features, hidden_size),
                activation,
                torch.nn.Linear(hidden_size, n_classes),
            ]
        )

        self.model = torch.nn.Sequential(*module_list)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(spectrogram)
