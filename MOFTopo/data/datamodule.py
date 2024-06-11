from lightning import LightningDataModule

from torch.utils.data import random_split, DataLoader
from torch import Generator

from hydra.utils import instantiate

from MOFTopo.common.data_utils import parse_zeolites
from MOFTopo.data.dataset import ZeoliteDataset


class BaseDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()


class ZeoliteDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        """
        Hyperparameters:
            train_pctg
            val_pctg
            test_pctg
            batch_size

            dataset - Recursive definition of the whole dataset
        """
        super().__init__(*args, **kwargs)
    
    def prepare_data(self,):
        if self.hparams.raw_data_dir is not None:
            parse_zeolites(self.hparams.raw_data_dir, self.hparams.parsed_data_dir)
    
    def setup(self, stage):
        zeolite_dataset = instantiate(self.hparams.dataset)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset=zeolite_dataset,
            lengths=[self.hparams.train_pctg, self.hparams.val_pctg, self.hparams.test_pctg],
            generator=Generator().manual_seed(0),
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)


