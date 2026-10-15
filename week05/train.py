"""Fine-tune the offline model into a streaming one.

python train.py --init base.pt --out stream.pt --chunk 8 --left 128

Budget (do not exceed in the graded run): --steps 1000 x --batch_sec 200 s of audio (~55 h).
"""

import argparse
import math
import time

import torch
import torchaudio

from src.data import ASRDataset, DurationBatchSampler
from src.features import SpecAugment
from src.metrics import wer
from src.model import ASR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="directory with train.jsonl, dev.jsonl and the audio")
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, required=True, help="> 0, frames of 40 ms")
    ap.add_argument("--left", type=int, default=128, help="> 0, left context in frames")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_sec", type=float, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max_dur", type=float, default=20)
    # TODO bonus: knowledge distillation from the offline model
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    assert a.chunk > 0 and a.left > 0, "streaming model: chunk > 0 and left > 0"
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    amp = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16  # T4 (sm75): fp16
    model = ASR.load(a.init, dev).float().train()
    # TODO bonus: the offline model as a teacher
    train = ASRDataset(f"{a.data}/train.jsonl", tokenizer=model.tokenizer, max_dur=a.max_dur)
    sampler = DurationBatchSampler([r["duration"] for r in train.rows], a.batch_sec, shuffle=True, seed=a.seed)
    loader = torch.utils.data.DataLoader(train, batch_sampler=sampler, collate_fn=ASRDataset.collate, num_workers=4)
    spec_augment = SpecAugment()
    params = [p for n, p in model.named_parameters() if not n.startswith("features")]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / max(1, a.warmup)) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    scaler = torch.amp.GradScaler(init_scale=1024.0, enabled=amp == torch.float16)

    step, epoch, t0, audio_h = 0, 0, time.time(), 0.0
    while step < a.steps:
        sampler.set_epoch(epoch)
        for b in loader:
            x, n = b["wav"].to(dev), b["wav_len"].to(dev)
            y, y_len = b["tokens"].to(dev), b["tokens_len"].to(dev)
            with torch.no_grad():
                mel, mel_len = model.features(x, n)
                mel = spec_augment(mel)
            with torch.autocast("cuda", dtype=amp):
                enc, enc_len = model.encoder(mel, mel_len, a.chunk, a.left)
                logits = model.rnnt(enc, y)
            loss = torchaudio.functional.rnnt_loss(logits.float(), y.int(), enc_len.int(), y_len.int(),
                                                   blank=model.rnnt.blank, reduction="mean",
                                                   fused_log_softmax=False)  # the joint gives log-probs
            # TODO bonus: extra losses (e.g. distillation) go here
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            audio_h += float(n.sum()) / 16000 / 3600
            if step % 100 == 0 or step == a.steps:
                print(f"step {step} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} "
                      f"audio {audio_h:.1f} h  {time.time() - t0:.0f} s", flush=True)
            if step >= a.steps:
                break
        epoch += 1

    model.stream = {"chunk": a.chunk, "left": a.left}
    model.save(a.out)
    model.eval()
    dev_set = ASRDataset(f"{a.data}/dev.jsonl")
    hyps = model.transcribe(dev_set, a.chunk, a.left)
    print(f"dev WER {100 * wer([r['text'] for r in dev_set.rows], hyps):.2f}%  (chunk {a.chunk}, left {a.left})")


if __name__ == "__main__":
    main()
