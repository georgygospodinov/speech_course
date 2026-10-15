"""Text normalization and the character tokenizer."""

import re


def normalize(text, for_wer=False):
    t = text.lower().replace("ё", "е")
    t = re.sub(r"(?<=[а-я])[-'’](?=[а-я])", "", t)  # "кто-то" -> "ктото"
    t = re.sub(r"[^а-я ]+", " ", t)
    if for_wer:
        t = t.replace("й", "и")  # Golos transcripts partly lost "й"
    return " ".join(t.split())


class CharTokenizer:
    """Characters <-> ids: vocab = space + 32 lowercase Cyrillic letters (the heads use id len(vocab) as blank)."""

    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.ids = {c: i for i, c in enumerate(self.vocab)}

    def encode(self, text):
        return [self.ids[c] for c in normalize(text)]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)
