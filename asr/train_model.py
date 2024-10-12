import logging

import hydra
import omegaconf
import torch
import pytorch_lightning as pl

<<<<<<< HEAD
from src.models import CTCModel
=======
from src.models import CTCModel, LASModel
>>>>>>> week4_seminar


MODEL_CLASSES = {
    "ctc_model": CTCModel,
<<<<<<< HEAD
=======
    "las_model": LASModel,
>>>>>>> week4_seminar
}


def resolve_model_class(model_class_name: str):
    if model_class_name in MODEL_CLASSES:
        return MODEL_CLASSES[model_class_name]
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")


@hydra.main(config_path="conf", config_name="conformer_ctc")
def main(conf: omegaconf.DictConfig) -> None:
    model = resolve_model_class(conf.model.model_class)(conf)

    if conf.get("init_weights", False):
        ckpt = torch.load(conf.init_weights, map_location="cpu")
<<<<<<< HEAD
        if 'state_dict' in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt)
=======
        model.load_state_dict(ckpt["state_dict"])
>>>>>>> week4_seminar
        logging.getLogger("lightning").info("successful load initial weights")

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(save_dir="logs"), **conf.trainer
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
