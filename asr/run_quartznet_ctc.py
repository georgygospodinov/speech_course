import logging

import hydra
import omegaconf
import torch
import pytorch_lightning as pl

from src.models import QuartzNetCTC


@hydra.main(config_path="conf", config_name="quarznet_5x5_ru")
def main(conf: omegaconf.DictConfig) -> None:
    model = QuartzNetCTC(conf)

    if conf.model.init_weights:
        ckpt = torch.load(conf.model.init_weights, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        logging.getLogger("lightning").info("successful load initial weights")

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(save_dir="logs"), **conf.trainer
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
