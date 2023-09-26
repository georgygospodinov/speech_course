import collections
import sys
from typing import Deque, List

import hydra
import numpy as np
import omegaconf
import pyaudio
import torch

from record import create_stream, record_audio


def audio_bytes_to_tensor(frames: bytes) -> torch.Tensor:
    audio_frame = np.frombuffer(frames, dtype=np.int16)
    float_audio_frame = audio_frame / (1 << 15)
    return torch.from_numpy(float_audio_frame).unsqueeze(0).to(torch.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


@torch.no_grad()
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: omegaconf.DictConfig) -> None:

    feature_extractor = hydra.utils.instantiate(conf.feature_extractor)

    model = hydra.utils.instantiate(conf.model)

    cfg = conf.inference

    buffer_size = int(cfg.window_size_seconds / cfg.window_shift_seconds)

    spec_frames: Deque[torch.Tensor] = collections.deque(maxlen=buffer_size)

    last_probs = collections.deque(
        [
            np.ones(len(cfg.idx_to_keyword)) / len(cfg.idx_to_keyword)
            for _ in range(cfg.avg_window_size)
        ],
        maxlen=cfg.avg_window_size,
    )

    pa_manager = pyaudio.PyAudio()
    stream = create_stream(
        pa_manager,
        sample_rate=conf.sample_rate,
        frames_per_buffer=int(conf.sample_rate * cfg.window_shift_seconds),
    )

    while True:

        try:
            byte_audio_frame = record_audio(
                stream, duration_seconds=cfg.window_shift_seconds
            )
            spec_frames.append(
                feature_extractor(audio_bytes_to_tensor(byte_audio_frame))
            )

            if len(spec_frames) == buffer_size:
                spectrogram = torch.cat(list(spec_frames), dim=2).numpy()
                logits: List[np.ndarray] = model.run(
                    ["logits"], {"features": spectrogram}
                )
                last_probs.append(
                    softmax(logits[0][0])
                )
                averaged_probs: np.ndarray = sum(last_probs) / len(last_probs)
                argmax_id = int(averaged_probs.argmax())
                keyword = cfg.idx_to_keyword[argmax_id]
                keyword_proba = averaged_probs[argmax_id]
                print(
                    keyword
                    if keyword_proba > cfg.threshold
                    and keyword != cfg.idx_to_keyword[-1]
                    else ""
                )

        except KeyboardInterrupt:
            pa_manager.terminate()
            stream.stop_stream()
            stream.close()
            sys.exit(0)


if __name__ == "__main__":
    main()
