"""Cache-based streaming inference (homework, part 2).

    streamer = Streamer(model, chunk=8, left=128)
    state = streamer.init_state()
    for piece in audio_pieces:                  # any lengths, e.g. 10 ms
        tokens, enc, state = streamer.step(piece, state)
    tokens, enc, state = streamer.flush(state)  # end of stream

step() returns the tokens emitted after this piece of audio and the encoder frames computed
(enc [n, 768]; check.py compares them with model.encode(..., chunk, left) and decodes them itself).
Everything the engine remembers must be in `state` (tensors, numbers, lists of them): check.py creates a
new Streamer object for every call, and `state` must not grow with the length of the stream.
"""

import torch
import torch.nn.functional as F


class Streamer:
    def __init__(self, model, chunk: int, left: int):
        assert chunk > 0 and left > 0, "streaming needs chunk >= 1 and left >= 1 frames"
        self.m, self.chunk, self.left = model, chunk, left

    def init_state(self):
        """State of a new stream. You may change it, but keep everything the engine remembers here."""
        enc, dev = self.m.encoder, next(self.m.parameters()).device
        d = enc.layers[0].norm_out.normalized_shape[0]
        return {
            "audio": torch.zeros(0, device=dev),        # samples not yet used by a whole mel frame
            "sub1": torch.zeros(64, 2, device=dev),     # inputs of subsampling conv 1 (2 = left zero padding)
            "sub2": torch.zeros(d, 2, device=dev),      # inputs of subsampling conv 2
            "frames": torch.zeros(0, d, device=dev),    # encoder-input frames waiting for a full chunk
            "pos": 0,                                   # position of the next chunk (rotary)
            "kv": [None] * len(enc.layers),             # per layer: K, V of the last `left` frames
            "conv": [None] * len(enc.layers),           # per layer: last k//2 depthwise-conv inputs
            "dec": None,                                # RNN-T decoder state (model.rnnt.greedy)
        }

    def step(self, audio, state):
        """audio: 1-d float tensor (16 kHz) of any length -> (new tokens: list[int], enc [n, 768], state).

        The pipeline; each stage keeps in `state` what the next call will still need:
        1. mel (model.features.mel + log as in LogMel): frame m = samples [160 m, 160 m + 320); windows
           overlap, so keep the samples that are not yet covered by a whole frame.
        2. subsampling (model.encoder.pre_encode.conv[0], [2], each + ReLU): conv k=5, stride 2, padding 2, output j
           needs inputs 2j-2 .. 2j+2. At the stream start the left padding is zeros; zeros on the right only in flush().
        3. encoder: wait for a full chunk of `chunk` frames, then repeat ConformerLayer.forward with caches:
           attention (self_attn.qkv / attend) over [K, V of at most `left` previous frames, the chunk], positions
           continue across chunks; depthwise conv (conv.pre / post) with the k//2 previous inputs and zeros after
           the chunk, as your depthwise_conv does; then model.encoder.out_proj.
        4. RNN-T greedy over the new frames: model.rnnt.greedy(enc, enc_len, state) continues from the decoder
           state of the previous call.
        """
        # TODO 3
        raise NotImplementedError("TODO 3")

    def flush(self, state):
        """End of stream: finish the last, possibly incomplete, chunk (zeros on the right of the subsampling,
        as in the offline model) -> (tokens, enc, state of a new stream)."""
        # TODO 3
        raise NotImplementedError("TODO 3")
