"""The dataset and duration-bucketed batches."""

import json
import os
import random

import soundfile as sf
import torch


class ASRDataset(torch.utils.data.Dataset):
    """Utterances of a manifest: item = {"idx", "row", "wav" [N] (16 kHz), "tokens" [U] (if a tokenizer is given)}.

    manifest: jsonl, one utterance per line: {"audio": path relative to audio_root, "duration": s, "text": ...,
    ["words": [{"text", "start", "end"}]]}; audio_root defaults to the manifest's directory.
    """

    def __init__(self, manifest, audio_root=None, tokenizer=None, max_dur=None, limit=None):
        with open(manifest) as f:
            rows = [json.loads(line) for line in f]
        self.rows = [r for r in rows if max_dur is None or r["duration"] <= max_dur][:limit]
        self.audio_root = os.path.dirname(manifest) if audio_root is None else audio_root
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        wav, sr = sf.read(os.path.join(self.audio_root, row["audio"]), dtype="float32")
        assert sr == 16000
        item = {"idx": i, "row": row, "wav": torch.from_numpy(wav)}
        if self.tokenizer is not None:
            item["tokens"] = torch.tensor(self.tokenizer.encode(row["text"]))
        return item

    @staticmethod
    def collate(items):
        """-> {"idx", "rows", "wav" [B, N] zero-padded, "wav_len" [B], and "tokens" [B, U], "tokens_len" [B]}."""
        pad = lambda xs: torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)  # noqa: E731
        batch = {"idx": [it["idx"] for it in items], "rows": [it["row"] for it in items],
                 "wav": pad([it["wav"] for it in items]), "wav_len": torch.tensor([len(it["wav"]) for it in items])}
        if "tokens" in items[0]:
            batch["tokens"] = pad([it["tokens"] for it in items])
            batch["tokens_len"] = torch.tensor([len(it["tokens"]) for it in items])
        return batch


class DurationBatchSampler(torch.utils.data.Sampler):
    """Batches of utterances of similar duration, padded duration <= batch_sec. With shuffle, the batch order
    depends on seed + epoch: call set_epoch() before every epoch (as with DistributedSampler)."""

    def __init__(self, durations, batch_sec, shuffle=False, seed=0):
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.batches, cur = [], []
        for i in sorted(range(len(durations)), key=lambda i: durations[i]):
            if cur and durations[i] * (len(cur) + 1) > batch_sec:
                self.batches.append(cur)
                cur = []
            cur.append(i)
        self.batches.append(cur)

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        return iter(batches)
