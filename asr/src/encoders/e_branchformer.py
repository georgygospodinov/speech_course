from typing import Optional, Union

from omegaconf import DictConfig
import hydra
import torch
import torch.nn

from src.submodules.subsampling import StackingSubsampling
from src.submodules.positional_encoding import PositionalEncoding


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_linear_after_conv: bool = False,
    ):
        """
        Convolutional Spatial Gating Unit (https://arxiv.org/pdf/2207.02971)
        Args:
            size: int - Input embedding dim
            kernel_size: int - Kernel size in DepthWise Conv
            dropout: float - Dropout rate
            use_linear_after_conv: bool - Whether to use linear layer after convolution
        """

        super().__init__()
        # TODO: LayerNorm
        self.norm = None

        # TODO: DepthWise Conv
        self.conv = None

        if use_linear_after_conv:
            self.linear = None
        else:
            self.linear = None

        # Dropout
        self.dropout = None

    def forward(self, x: torch.Tensor):
        """
        Inputs:
            x: B x T x C
        Outputs:
            out: B x T x C
        """
        # TODO

        return None


class ConvolutionalGatingMLP(torch.nn.Module):
    def __init__(
        self,
        size: int,
        kernel_size: int,
        expansion_factor: int = 6,
        dropout: float = 0.0,
        use_linear_after_conv: bool = False,
    ):
        """
        Convolutional Gating MLP (https://arxiv.org/pdf/2207.02971)
        Args:
            size: int - Input embedding dim
            kernel_size: int - Kernel size for DepthWise Conv in ConvolutionalSpatialGatingUnit
            expansion_factor: int - Dim expansion factor for ConvolutionalSpatialGatingUnit
            dropout: float - Dropout rate
            use_linear_after_conv: bool - Whether to use linear layer after convolution
        """
        super().__init__()

        # TODO: First Channel Projection with GeLU Activation
        self.channel_proj1 = None

        # TODO: Convlutional Spatial Gating Unit
        self.csgu = None

        # TODO: Second Channel Projection with GeLU Activation
        self.channel_proj2 = None

    def forward(self, features: torch.Tensor):
        """
        Inputs:
            features: B x T x C
        Outputs:
            out: B x T x C

        """
        # TODO

        return None


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        """
        Standard FeedForward layer from Transformer block,
        consisting of a compression and decompression projection
        with an activation function.
        Args:
            input_dim: int - Input embedding dim
            hidden_dim: int - Hidden dim
            dropout: float - Dropout rate
            activation: torch.nn.Module - Activation function
        """
        super().__init__()
        self.linear1 = None
        self.linear2 = None
        self.dropout = None
        self.activation = None

    def forward(self, features: torch.Tensor):
        """
        Inputs:
            features: B x T x C
        Outputs:
            out: B x T x C
        """
        # TODO

        return None


class EBranchformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        size: int,
        attn_config: Union[DictConfig, dict],
        cgmlp_config: Union[DictConfig, dict],
        ffn_expansion_factor: int = 4,
        droupout: float = 0.0,
        merge_conv_kernel: int = 3,
    ):
        """
        E-Bbranchformer Layer (https://arxiv.org/pdf/2210.00077)
        Args:
            size: int - Embedding dim
            attn_config: DictConfig or dict - Config for MultiheadAttention
            cgmlp_config: DictConfig or dict - Config for ConvolutionalGatingMLP
            ffn_expansion_factor: int - Expansion factor for FeedForward
            dropout: float - Dropout rate
            merge_conv_kernel: int - Kernel size for merging module
        """

        super().__init__()

        # MultiheadAttention from torch.nn
        self.attn = None

        # ConvolutionalGatingMLP module
        self.cgmlp = None

        # First and Second FeedForward modules
        self.feed_forward1 = None
        self.feed_forward2 = None

        # Normalization modules
        self.norm_ffn1 = None
        self.norm_ffn2 = None
        self.norm_mha = None
        self.norm_mlp = None
        self.norm_final = None

        self.dropout = None

        # DepthWise Convolution and Linear projection for merging module
        self.depthwise_conv_fusion = None
        self.merge_proj = None

    def forward(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
    ):
        """
        Inputs:
            features: B x T x C
            features_length: B
            pos_emb: B x T x C - Optional
        Outputs:
            out: B x T x C
        """
        # TODO

        return None


class EBranchformerEncoder(torch.nn.Module):
    def __init__(
        self,
        subsampling_stride: int,
        features_num: int,
        d_model: int,
        layers_num: int,
        attn_config: Union[DictConfig, dict],
        cgmlp_config: Union[DictConfig, dict],
        ffn_expansion_factor: int = 2,
        dropout: float = 0.0,
        merge_conv_kernel: int = 3,
    ):
        super().__init__()
        self.subsampling = StackingSubsampling(
            stride=subsampling_stride, feat_in=features_num, feat_out=d_model
        )
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        self.layers = torch.nn.ModuleList()
        for _ in range(layers_num):
            layer = EBranchformerEncoderLayer(
                size=d_model,
                attn_config=attn_config,
                cgmlp_config=cgmlp_config,
                ffn_expansion_factor=ffn_expansion_factor,
                droupout=dropout,
                merge_conv_kernel=merge_conv_kernel,
            )
            self.layers.append(layer)

    def forward(self, features: torch.Tensor, features_length: torch.Tensor):
        features = features.transpose(1, 2)  # B x D x T -> B x T x D
        features, features_length = self.subsampling(features, features_length)
        features = self.pos_embedding(features)
        for layer in self.layers:
            features = layer(features, features_length)

        return features, features_length
