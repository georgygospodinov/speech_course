"""Log-mel features and SpecAugment."""

import torch
import torchaudio
from torch import nn


class LogMel(nn.Module):
    """Log-mel, 64 bins, 20 ms window, 10 ms hop, no centering: mel frame m = audio [160 m, 160 m + 320)."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=320, win_length=320, hop_length=160, n_mels=64, center=False)

    def forward(self, wav, wav_len):
        """wav [B, samples] -> log-mel [B, 64, T_mel], lengths."""
        return self.mel(wav).clamp(1e-9, 1e9).log(), (wav_len - 320) // 160 + 1


class SpecAugment(nn.Module):
    """Zeroes random frequency bands and time spans of a log-mel batch [B, 64, T] (training only)."""

    def __init__(self, n_freq=2, w_freq=15, n_time=2, w_time=0.05):
        super().__init__()
        self.n_freq, self.w_freq, self.n_time, self.w_time = n_freq, w_freq, n_time, w_time

    def forward(self, mel):
        B, Fq, T = mel.shape
        dev = mel.device
        f, t = torch.arange(Fq, device=dev), torch.arange(T, device=dev)
        mask = torch.zeros(B, Fq, T, dtype=torch.bool, device=dev)
        for _ in range(self.n_freq):
            w = torch.randint(0, self.w_freq + 1, (B, 1), device=dev)
            f0 = (torch.rand(B, 1, device=dev) * (Fq - w)).long()
            mask |= ((f >= f0) & (f < f0 + w))[:, :, None]
        for _ in range(self.n_time):
            w = (torch.rand(B, 1, device=dev) * self.w_time * T).long()
            t0 = (torch.rand(B, 1, device=dev) * (T - w)).long()
            mask |= ((t >= t0) & (t < t0 + w))[:, None, :]
        return mel.masked_fill(mask, 0.0)
