"""The whole ASR model: wav -> log-mel -> Conformer encoder -> RNN-T (+ CTC head, used in the seminar)."""

import torch
from torch import nn

from .conformer import Encoder
from .data import ASRDataset, DurationBatchSampler
from .decoders import CTCHead, RNNTDecoder
from .features import LogMel
from .text import CharTokenizer


class ASR(nn.Module):
    def __init__(self, vocab, d=384, layers=16, heads=8):
        super().__init__()
        self.tokenizer = CharTokenizer(vocab)
        self.features = LogMel()
        self.encoder = Encoder(d, layers, heads)
        self.rnnt, self.ctc = RNNTDecoder(len(vocab)), CTCHead(len(vocab))
        self.stream = {"chunk": 0, "left": 0}  # the streaming setup the model was trained for

    def encode(self, wav, wav_len, chunk=0, left=0):
        """Masked (chunked) forward: wav [B, N] -> encoder frames [B, T, 768], lengths. chunk = 0: offline."""
        mel, n = self.features(wav, wav_len)
        return self.encoder(mel, n, chunk, left)

    @torch.no_grad()
    def transcribe(self, dataset, chunk=0, left=0, batch_sec=300):
        """Masked forward + RNN-T greedy decoding of an ASRDataset -> hypotheses in the dataset order."""
        dev = next(self.parameters()).device
        sampler = DurationBatchSampler([r["duration"] for r in dataset.rows], batch_sec)
        hyps = [None] * len(dataset)
        for b in torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=ASRDataset.collate):
            mel, mel_len = self.features(b["wav"].to(dev), b["wav_len"].to(dev))  # fp32: fp16 STFT power overflows
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                enc, enc_len = self.encoder(mel, mel_len, chunk, left)
            for i, (tokens, _) in zip(b["idx"], self.rnnt.greedy(enc.float(), enc_len)[0]):
                hyps[i] = self.tokenizer.decode(tokens)
        return hyps

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu")
        model = cls(ck["vocab"], **ck["arch"])
        model.load_state_dict(ck["state_dict"])
        model.stream = ck["stream"]
        return model.to(device).eval()

    def save(self, path):
        layer = self.encoder.layers[0]
        arch = {"d": layer.norm_out.normalized_shape[0], "layers": len(self.encoder.layers), "heads": layer.self_attn.h}
        torch.save({"vocab": self.tokenizer.vocab, "arch": arch, "stream": self.stream,
                    "state_dict": self.state_dict()}, path)
