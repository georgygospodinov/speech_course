"""Output heads on top of the encoder: RNN-T (predictor + joint, greedy search) and CTC."""

import torch
from torch import nn


class Predictor(nn.Module):
    """RNN-T prediction network: LSTM over the previously emitted tokens."""

    def __init__(self, vocab_size, h=320):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, h, padding_idx=vocab_size)
        self.lstm = nn.LSTM(h, h)

    def forward(self, tokens, state=None):
        """One step: tokens [B, 1] (None = start: zero input) -> [B, 1, h], new LSTM state."""
        if tokens is None:
            emb = torch.zeros(state[0].size(1) if state else 1, 1, self.lstm.hidden_size,
                              device=self.embed.weight.device)
        else:
            emb = self.embed(tokens)
        g, state = self.lstm(emb.transpose(0, 1), state)
        return g.transpose(0, 1), state

    def sequence(self, tokens):
        """Training: outputs for the prefixes [start, y1, ..., yU] -> [B, U + 1, h]."""
        emb = torch.cat([torch.zeros_like(self.embed(tokens[:, :1])), self.embed(tokens)], 1)
        return self.lstm(emb.transpose(0, 1))[0].transpose(0, 1)


class Joint(nn.Module):
    def __init__(self, vocab_size, enc=768, pred=320, h=320):
        super().__init__()
        self.enc, self.pred = nn.Linear(enc, h), nn.Linear(pred, h)
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Linear(h, vocab_size + 1))

    def forward(self, enc, pred):
        """enc [B, T, 768], pred [B, U, 320] -> log-probs [B, T, U, V+1] (last = blank)."""
        return self.joint_net(self.enc(enc)[:, :, None] + self.pred(pred)[:, None]).log_softmax(-1)


class RNNTDecoder(nn.Module):
    """RNN-T head: predictor + joint; blank = vocab_size."""

    def __init__(self, vocab_size, enc=768, h=320):
        super().__init__()
        self.blank = vocab_size
        self.predictor, self.joint = Predictor(vocab_size, h), Joint(vocab_size, enc, h, h)

    def forward(self, enc, tokens):
        """Training: enc [B, T, 768], tokens [B, U] -> log-probs [B, T, U + 1, V + 1] for rnnt_loss."""
        return self.joint(enc, self.predictor.sequence(tokens))

    @torch.no_grad()
    def greedy(self, enc, enc_len, state=None, max_symbols=10):
        """Batched greedy search over enc [B, T, 768], enc_len [B] (tensor). `state` (predictor output and LSTM state) continues the search
        from a previous call on the preceding frames of the same streams; None = start of the streams.
        Returns per utterance (tokens, frames: the frame each token was emitted at) and the new state."""
        B, T, _ = enc.shape
        if state is None:
            h = self.predictor.lstm.hidden_size
            state = self.predictor(None, (torch.zeros(1, B, h, device=enc.device),) * 2)
        g, s = state
        out = [([], []) for _ in range(B)]
        for t in range(T):
            active = t < enc_len
            for _ in range(max_symbols):
                tok = self.joint(enc[:, t:t + 1], g)[:, 0, 0].argmax(-1)  # [B]
                emit = active & (tok != self.blank)
                if not emit.any():
                    break
                for b in emit.nonzero()[:, 0].tolist():
                    out[b][0].append(int(tok[b]))
                    out[b][1].append(t)
                g_new, s_new = self.predictor(tok[:, None], s)  # rows that output blank are discarded below
                e = emit[:, None, None].float()  # only utterances that emitted move their predictor
                g = e * g_new + (1 - e) * g
                s = tuple(e.transpose(0, 1) * s1 + (1 - e.transpose(0, 1)) * s0 for s1, s0 in zip(s_new, s))
        return out, (g, s)


class CTCHead(nn.Module):
    """CTC head (used in the seminar); blank = vocab_size."""

    def __init__(self, vocab_size, enc=768):
        super().__init__()
        self.blank = vocab_size
        self.linear = nn.Linear(enc, vocab_size + 1)

    def forward(self, enc):
        """enc [B, T, 768] -> log-probs [B, T, V + 1]."""
        return self.linear(enc).log_softmax(-1)

    def greedy(self, log_probs, lengths):
        """log_probs [B, T, V + 1], lengths [B] (tensor): best token per frame, merge repeats, drop blanks -> token lists."""
        out = []
        for lp, n in zip(log_probs.argmax(-1), lengths):
            out.append([int(t) for t in torch.unique_consecutive(lp[:n]) if t != self.blank])
        return out
