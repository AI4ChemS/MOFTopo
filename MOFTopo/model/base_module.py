from lightning import LightningModule
from hydra.utils import instantiate


class BaseModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
    
    def configure_optimizers(self,):
        print("CONFIGURING")
        print(self.hparams.optim.optimizer)
        optimizer = instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(),
        )

        if not self.hparams.optim.use_lr_scheduler:
            return [optim]
        
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler.scheduler, 
            optimizer=optim
        )

        return {
            "optimizer": 
            optim, 
            "lr_scheduler": 
            scheduler, 
            "monitor": self.hparams.optim.lr_scheduler.lr_monitor
        }

