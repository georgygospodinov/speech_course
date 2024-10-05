import torch
from torch import nn
import torchaudio

from src.submodules.subsampling import StackingSubsampling


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        subsampling_stride: int,
        features_num: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_exp_factor: int,
        kernel_size: int,
        dropout: int,
    ):
        super().__init__()
        self.subsampling = StackingSubsampling(
            stride=subsampling_stride, feat_in=features_num, feat_out=d_model
        )
        self.backbone = torchaudio.models.Conformer(
            input_dim=d_model,
            num_heads=n_heads,
            ffn_dim=d_model * ff_exp_factor,
            num_layers=n_layers,
            depthwise_conv_kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor, features_length: torch.Tensor):
        features = features.transpose(1, 2)  # B x D x T -> B x T x D
        features, features_length = self.subsampling(features, features_length)
        encoded, encoded_len = self.backbone(features, features_length)
        return encoded, encoded_len
