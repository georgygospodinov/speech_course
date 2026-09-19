import hydra
import omegaconf
from pytorch_lightning import seed_everything

from src.module import KWS
from utils import omegaconf_extension


@omegaconf_extension
@hydra.main(version_base="1.2", config_path="conf", config_name="conv1d.yaml")
def main(conf: omegaconf.DictConfig) -> None:
    seed_everything(314, workers=True)
    module = KWS(conf)
    logger = hydra.utils.instantiate(conf.logger)
    trainer = hydra.utils.instantiate(conf.trainer, logger=logger)
    trainer.fit(module)


if __name__ == "__main__":
    main()
