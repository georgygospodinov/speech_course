"""What makes the encoder streaming: the attention mask and the time convolution (homework, part 1).

Encoder frames are 40 ms. With chunk = C, frame t may only use frames up to the end of its chunk
(and at most `left` frames before the start of the chunk); chunk = 0 is the offline model.
Frame t covers audio [0.04 t, 0.04 t + 0.04) but, because of the subsampling convolutions and the mel window,
needs audio up to 0.04 t + 0.08 s, i.e. 40 ms after its own end.
"""

import torch
import torch.nn.functional as F
from torch import nn


def attention_mask(T: int, chunk: int, left: int, device=None) -> torch.Tensor:
    """[T, T] bool, True = query t may attend to key s. chunk >= 1 and left >= 1 (chunk = 0: offline)."""
    if chunk == 0:
        return torch.ones(T, T, dtype=torch.bool, device=device)
    # TODO 1: chunk mask with limited left context
    raise NotImplementedError("TODO 1")


def depthwise_conv(x: torch.Tensor, conv: nn.Conv1d, chunk: int) -> torch.Tensor:
    """Depthwise conv over time, x [B, d, T]. Offline (chunk = 0): symmetric 'same' conv."""
    k = conv.kernel_size[0]
    if chunk == 0:
        return F.conv1d(x, conv.weight, conv.bias, padding=k // 2, groups=conv.groups)
    # TODO 2: the conv must not see frames after the end of the current chunk
    raise NotImplementedError("TODO 2")
