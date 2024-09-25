import os

import hydra
import omegaconf
import torch

from src.module import KWS
from utils import omegaconf_extension


@omegaconf_extension
@hydra.main(version_base="1.2", config_path="conf", config_name="conv1d.yaml")
def main(conf: omegaconf.DictConfig) -> None:

    os.chdir(hydra.utils.get_original_cwd())

    model = KWS(conf)

    if conf.init_weights:
        ckpt = torch.load(conf.init_weights, map_location="cpu")
        model.load_state_dict(
            {k: v for k, v in ckpt["state_dict"].items() if "total" not in k}
        )

    features_params = conf.train_dataloader.dataset.transforms[0]

    torch.onnx.export(
        model,
        args=(
            torch.randn(
                1,
                features_params.n_mels,
                features_params.sample_rate // features_params.hop_length + 1,
            ),
        ),
        f="./data/kws.onnx",
        opset_version=14,
        input_names=["features"],
        output_names=["logprobs", "predictions"],
        dynamic_axes={
            "features": {0: "batch_size", 2: "time"},
            "logprobs": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    main()
