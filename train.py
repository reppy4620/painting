import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint

from data_modules import ImageDataModule
from modules.callback import GenerateImageCallback


@hydra.main(config_path='configs', config_name='config.yaml')
def main(cfg):
    pl.seed_everything(cfg.seed)

    model = instantiate(cfg.method.model)
    dm = ImageDataModule(cfg.method.data)
    dm.setup()

    wandb_logger = instantiate(cfg.logger)
    wandb_logger.watch(model, log='gradients', log_freq=100)

    model_checkpoint = ModelCheckpoint(
        filename=f'{cfg.method.name}' + '{epoch:04d}-{val_loss:.4f}',
        dirpath=f'saved_models',
        monitor='val_loss',
        save_top_k=5,
        save_last=True,
    )
    image_callback = GenerateImageCallback(
        val_images=next(iter(dm.val_dataloader()))[0],
        save_dir='generated_images'
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[model_checkpoint, image_callback],
        deterministic=True,
        **cfg.trainer
    )
    trainer.fit(model=model, datamodule=dm)


if __name__ == '__main__':
    main()
