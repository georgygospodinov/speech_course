"""Checks a streaming submission and computes the metrics used for grading.

python check.py --ckpt stream.pt [--n_latency 500]

1. causality : model.encode(..., chunk, left) must not use audio after the moment the chunk is complete
               (a 60 s stream, the audio after a cut replaced by noise; cuts every 1.37 s)
2. streamer  : Streamer (a new object on every call, so everything it remembers is in `state`), fed with
               10 ms (every 4th stream: random) pieces, returns the same encoder frames as model.encode (<= 1e-3) and the same
               tokens as greedy decoding of these frames; `state` does not grow with the stream length
3. latency   : tokens are decoded from the frames the Streamer returned; a token's time is how much audio had
               been fed when its frame came out; latency of a correct word = time of its last letter - its
               end in the forced alignment; mean over the streams of item 2 fed with 10 ms pieces (golos and 2 long
               streams). Word times: Montreal Forced Aligner
               (HF dataset govnejri/golos_mfa_punctuation), automatic, not checked by hand
4. WER       : model.encode + greedy decoding; the grade is the mean over golos, farfield, long
Grading runs with your src/streaming.py and src/streamer.py and our copies of the other files.
"""

import argparse
import itertools
import json
import random

import numpy as np
import torch

from src.data import ASRDataset
from src.metrics import wer, word_latencies
from src.model import ASR
from src.streamer import Streamer

FRAME_SEC, LOOKAHEAD_SEC = 0.04, 0.08  # frame t needs audio up to 0.04 * t + 0.08 s
PIECE = 160  # 10 ms
MAX_LATENCY = 0.300
STATE_AT = 40.0  # s: state size there vs at the end of a 60 s stream


def numel(o):
    if torch.is_tensor(o):
        return o.numel()
    if isinstance(o, np.ndarray):
        return o.size
    if isinstance(o, dict):
        return sum(numel(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(numel(v) for v in o)
    return 0 if o is None else 1


def stream(model, wav, chunk, left, random_pieces=False, seed=0):
    """Returns the Streamer's tokens, encoder frames [T, 768], the audio time (s) when each frame came out,
    and state sizes {STATE_AT: ..., "end": ...}."""
    rng, dev = random.Random(seed), wav.device
    state = Streamer(model, chunk, left).init_state()
    tokens, encs, ftimes, sizes, i = [], [], [], {}, 0
    while True:
        last = i >= len(wav)
        if last:
            sizes["end"] = numel(state)  # before flush: the stream is over after it
        k = rng.randint(1, 4000) if random_pieces else PIECE
        st = Streamer(model, chunk, left)  # fresh object: all memory must be in `state`
        tok, enc, state = st.flush(state) if last else st.step(wav[i:i + k], state)
        i = min(i + k, len(wav))
        enc = torch.as_tensor(enc, dtype=torch.float32, device=dev).reshape(-1, 768)
        tokens += list(tok)
        encs.append(enc)
        ftimes += [i / 16000] * len(enc)
        if STATE_AT not in sizes and i / 16000 >= STATE_AT:
            sizes[STATE_AT] = numel(state)
        if last:
            break
    return tokens, torch.cat(encs), ftimes, sizes


def check_causality(model, chunk, left, stream, dev):
    wav = stream["wav"].to(dev)
    n = torch.tensor([len(wav)], device=dev)
    e1, _ = model.encode(wav[None], n, chunk, left)
    t_avail = FRAME_SEC * ((torch.arange(e1.size(1), device=dev) // chunk + 1) * chunk - 1) + LOOKAHEAD_SEC
    worst = 0.0
    for cut in np.arange(1.0, len(wav) / 16000, 1.37):
        w2 = wav.clone()
        w2[int(cut * 16000):] = torch.randn_like(w2[int(cut * 16000):]) * 0.1
        e2, _ = model.encode(w2[None], n, chunk, left)
        ready = t_avail <= cut
        if ready.any():
            worst = max(worst, (e1 - e2)[0, ready].abs().max().item())
    return worst


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data", help="directory with the test manifests and the audio")
    ap.add_argument("--sets", nargs="+", default=["golos", "farfield", "long"])
    ap.add_argument("--n_latency", type=int, default=500, help="golos utterances run through the Streamer")
    ap.add_argument("--out", default=None, help="save results as json")
    ap.add_argument("--hidden", action="store_true", help="organizers: use <data>/hidden/*.jsonl")
    a = ap.parse_args()

    def test_set(name, **kw):  # audio paths in all manifests are relative to the data directory
        return ASRDataset(f"{a.data}/{'hidden/' if a.hidden else ''}{name}.jsonl", audio_root=a.data, **kw)

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASR.load(a.ckpt, dev).float()
    C, L = model.stream["chunk"], model.stream["left"]
    assert C > 0 and 0 < L <= 1000, f"checkpoint has chunk={C}, left={L}: need chunk >= 1, 1 <= left <= 1000"
    res = {"chunk": C, "left": L}
    print(f"chunk {C} frames ({C * 40} ms), left context {L} frames")

    res["causality_max_diff"] = check_causality(model, C, L, test_set("long")[0], dev)
    print(f"1. causality: max change of frames that are complete before the cut {res['causality_max_diff']:.1e}")

    streams = [test_set("golos", limit=a.n_latency), test_set("long", limit=2)]
    n_streams = sum(len(s) for s in streams)
    diffs, same, lat, growth = [], 0, [], []
    for k, item in enumerate(itertools.chain(*streams)):
        r, wav = item["row"], item["wav"].to(dev)
        rnd = k % 4 == 3  # every 4th stream: random piece sizes (engine check only, not latency)
        tokens, enc, ftimes, sizes = stream(model, wav, C, L, random_pieces=rnd, seed=k)
        ref, n = model.encode(wav[None], torch.tensor([len(wav)], device=dev), C, L)
        diffs.append((enc - ref[0]).abs().max().item() if enc.shape == ref[0].shape else float("inf"))
        if enc.size(0) == 0:
            continue
        dec_tokens, frames = model.rnnt.greedy(enc[None], torch.tensor([enc.size(0)], device=dev))[0][0]
        same += tokens == dec_tokens
        if not rnd:
            lat += word_latencies(r, dec_tokens, [ftimes[f] for f in frames], model.tokenizer.vocab)
        if STATE_AT in sizes:
            growth.append(sizes["end"] / sizes[STATE_AT] if sizes[STATE_AT] else float("inf"))
    res["stream_max_diff"], res["same_tokens"] = max(diffs), same / n_streams
    res["state_growth"] = max(growth) if growth else float("inf")
    res["latency_mean"] = sum(lat) / len(lat) if lat else float("inf")
    print(f"2. streamer: max|enc - masked enc| {res['stream_max_diff']:.1e}, its tokens = greedy of its frames "
          f"in {same}/{n_streams} streams, state size end / at {STATE_AT:.0f} s = {res['state_growth']:.2f}")
    print(f"3. latency: mean {1000 * res['latency_mean']:.0f} ms over {len(lat)} words")

    res["wer"] = {}
    for s in a.sets:
        test = test_set(s)
        res["wer"][s] = wer([r["text"] for r in test.rows], model.transcribe(test, C, L))
        print(f"4. WER {s}: {100 * res['wer'][s]:.2f}%")
    res["wer_mean"] = sum(res["wer"].values()) / len(res["wer"])

    ok = {"causal": res["causality_max_diff"] < 1e-4,
          "streamer": res["stream_max_diff"] <= 1e-3 and res["same_tokens"] >= 0.95,
          "bounded state": res["state_growth"] <= 1.05,
          f"latency <= {MAX_LATENCY * 1000:.0f} ms": res["latency_mean"] <= MAX_LATENCY}
    res["gates"] = ok
    print("gates:", ", ".join(f"{k}: {'OK' if v else 'FAIL'}" for k, v in ok.items()))
    print(f"mean WER {100 * res['wer_mean']:.2f}%" + ("" if all(ok.values()) else "  (gates failed)"))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
