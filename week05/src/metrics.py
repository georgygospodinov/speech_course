"""WER and word latency."""

import numpy as np

from .text import normalize


def align(ref, hyp):
    """Word-level Levenshtein; returns list of (ref_index, hyp_index) pairs of exactly matched words."""
    D = np.arange(len(hyp) + 1)[None].repeat(len(ref) + 1, 0)
    D[:, 0] = np.arange(len(ref) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            D[i, j] = min(D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    i, j, pairs = len(ref), len(hyp), []
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return D[-1, -1], pairs[::-1]


def wer(refs, hyps):
    errors = words = 0
    for r, h in zip(refs, hyps):
        r, h = normalize(r, True).split(), normalize(h, True).split()
        errors += align(r, h)[0]
        words += len(r)
    return errors / max(words, 1)


def word_latencies(row, tokens, times, vocab):
    """Latency of each correctly recognised reference word = time when its last letter was
    emitted - its end in the forced alignment (row["words"]). tokens/times: emitted ids and the
    audio time (s) consumed when each was emitted."""
    hyp, cur_t, words = [], None, ""
    for tok, t in zip(tokens + [0], times + [None]):  # trailing space closes the last word
        if vocab[tok] == " ":
            if words:
                hyp.append((words, cur_t))
            words = ""
        else:
            words, cur_t = words + vocab[tok], t
    ref = [(normalize(w["text"], True), w["end"]) for w in row["words"]]
    ref = [(w, e) for w, e in ref if w and " " not in w]
    hyp = [(normalize(w, True), t) for w, t in hyp]
    _, pairs = align([w for w, _ in ref], [w for w, _ in hyp])
    return [hyp[j][1] - ref[i][1] for i, j in pairs]
