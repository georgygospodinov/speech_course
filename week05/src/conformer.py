"""Conformer encoder as in GigaAM v3: subsampling x4 (40 ms frames) -> layers -> projection to 768."""

import torch
import torch.nn.functional as F
from torch import nn

from .streaming import attention_mask, depthwise_conv


class Subsampling(nn.Module):
    """Two conv1d(k=5, stride=2): frame t sees mel frames 4t-6 .. 4t+6."""

    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(64, d, 5, 2, 2), nn.ReLU(), nn.Conv1d(d, d, 5, 2, 2), nn.ReLU())

    def forward(self, mel, mel_len):
        x, n = mel, mel_len
        x = x.masked_fill(torch.arange(x.size(2), device=x.device) >= n[:, None, None], 0.0)
        for m in self.conv:
            x = m(x)
            if isinstance(m, nn.Conv1d):
                n = (n - 1) // 2 + 1
                x = x.masked_fill(torch.arange(x.size(2), device=x.device) >= n[:, None, None], 0.0)
        return x.transpose(1, 2), n  # [B, T, d]


def rotary(x, pos, base=5000.0):
    """Rotary position embedding of q or k [B, H, T, dk] at positions `pos` [T]: q·k then depends only
    on the position difference."""
    dim = x.size(-1)
    freq = pos[:, None].float() / base ** (torch.arange(0, dim, 2, device=x.device).float() / dim)
    cos, sin = torch.cat([freq, freq], -1).cos(), torch.cat([freq, freq], -1).sin()  # [T, dk]
    x1, x2 = x.chunk(2, -1)
    return x * cos.to(x.dtype) + torch.cat([-x2, x1], -1) * sin.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.linear_q, self.linear_k = nn.Linear(d, d), nn.Linear(d, d)
        self.linear_v, self.linear_out = nn.Linear(d, d), nn.Linear(d, d)

    def qkv(self, x, pos):
        """x [B, T, d] -> q, k, v [B, H, T, dk]; q and k rotated by their positions `pos` [T]."""
        B, T, _ = x.shape
        split = lambda y: y.view(B, T, self.h, self.dk).transpose(1, 2)  # noqa: E731
        return rotary(split(self.linear_q(x)), pos), rotary(split(self.linear_k(x)), pos), split(self.linear_v(x))

    def attend(self, q, k, v, mask):
        """mask [B or 1, Tq, Tk] bool (True = allowed) -> [B, Tq, d]."""
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        return self.linear_out(y.transpose(1, 2).flatten(2))


class FeedForward(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class ConvModule(nn.Module):
    """pointwise conv + GLU -> depthwise conv over time (streaming.depthwise_conv) -> LayerNorm, SiLU, pointwise."""

    def __init__(self, d, k=5):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1)
        self.depthwise_conv = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.batch_norm = nn.LayerNorm(d)  # (sic) name from GigaAM
        self.pointwise_conv2 = nn.Conv1d(d, d, 1)

    def pre(self, x, pad):
        """x [B, T, d] -> depthwise conv input [B, d, T]."""
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        return x.masked_fill(pad[:, None], 0.0)

    def post(self, y):
        """depthwise conv output [B, d, T] -> [B, T, d]."""
        y = F.silu(self.batch_norm(y.transpose(1, 2)))
        return self.pointwise_conv2(y.transpose(1, 2)).transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm_feed_forward1, self.feed_forward1 = nn.LayerNorm(d), FeedForward(d)
        self.norm_self_att, self.self_attn = nn.LayerNorm(d), Attention(d, heads)
        self.norm_conv, self.conv = nn.LayerNorm(d), ConvModule(d)
        self.norm_feed_forward2, self.feed_forward2 = nn.LayerNorm(d), FeedForward(d)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x, pos, mask, pad, chunk):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        q, k, v = self.self_attn.qkv(self.norm_self_att(x), pos)
        x = x + self.self_attn.attend(q, k, v, mask)
        c = self.conv.pre(self.norm_conv(x), pad)
        x = x + self.conv.post(depthwise_conv(c, self.conv.depthwise_conv, chunk))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Encoder(nn.Module):
    def __init__(self, d=384, layers=16, heads=8, out_dim=768):
        super().__init__()
        self.pre_encode = Subsampling(d)
        self.layers = nn.ModuleList(ConformerLayer(d, heads) for _ in range(layers))
        self.out_proj = nn.Linear(d, out_dim)

    def forward(self, mel, mel_len, chunk=0, left=0):
        """log-mel [B, 64, T_mel] -> [B, T, 768], lengths. chunk / left: see streaming.py."""
        x, n = self.pre_encode(mel, mel_len)
        T = x.size(1)
        pad = torch.arange(T, device=x.device)[None] >= n[:, None]               # [B, T]
        mask = attention_mask(T, chunk, left, x.device)[None] & ~pad[:, None, :]  # [B, T, T]
        mask |= ~mask.any(-1, keepdim=True)  # padded queries: attend somewhere (output unused)
        pos = torch.arange(T, device=x.device)
        for layer in self.layers:
            x = layer(x, pos, mask, pad, chunk)
        return self.out_proj(x), n
