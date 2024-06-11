from os.path import join

from torch import manual_seed

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig

from MOFTopo import PROJECT_ROOT
from MOFTopo.model import GemNet

manual_seed(0)


@hydra.main(config_path=join(PROJECT_ROOT, "conf"), config_name="default_config", version_base=None)
def main(cfg: DictConfig):
    dataset = instantiate(
        cfg.data.dataset,
        _recursive_=False,
    )

    datamodule = instantiate(
        cfg.data.datamodule,
        dataset=dataset,
        _recursive_=False,
    )

    module = instantiate(
        cfg.model.model,
        optim=cfg.optim,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()