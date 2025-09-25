import onnxruntime as ort
import torch


class SpecScaler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


def load_model(
    path: str, intra_threads: int, inter_threads: int
) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_threads
    sess_options.inter_op_num_threads = inter_threads
    return ort.InferenceSession(path, sess_options=sess_options)
