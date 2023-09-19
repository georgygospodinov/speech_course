import os
from typing import List, Tuple

import hydra
import omegaconf
import pandas as pd
import torch

from src.module import KWS
from utils import omegaconf_extension


@omegaconf_extension
@hydra.main(version_base="1.2", config_path="conf", config_name="conv1d.yaml")
def main(conf: omegaconf.DictConfig) -> None:

    os.chdir(hydra.utils.get_original_cwd())

    module = KWS(conf)
    if conf.init_weights:
        ckpt = torch.load(conf.init_weights, map_location="cpu")
        module.load_state_dict(ckpt["state_dict"])

    logger = hydra.utils.instantiate(conf.logger)
    trainer = hydra.utils.instantiate(conf.trainer, logger=logger)

    predictions: List[Tuple[torch.Tensor, torch.Tensor]] = trainer.predict(
        module, return_predictions=True
    )

    ids_tensor, labels_tensor = zip(*predictions)
    ids = torch.cat(ids_tensor).numpy()
    labels = torch.cat(labels_tensor).numpy().tolist()

    df = pd.read_csv(conf.predict_dataloader.dataset.manifest_path).iloc[ids]
    df["label"] = [
        conf.predict_dataloader.dataset.idx_to_keyword[label] for label in labels
    ]
    df[["index", "label"]].to_csv("submit.csv", index=False)


if __name__ == "__main__":
    main()
