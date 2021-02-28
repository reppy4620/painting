import pathlib
import wandb
import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import save_image


class GenerateImageCallback(Callback):
    def __init__(self, val_images, save_dir):
        self.val_sketches = torch.stack(tuple(zip(*val_images))[0])
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        if len(self.val_sketches.size()) != 4:
            self.val_sketches = self.val_sketches.unsqueeze(1)
        color_gen = pl_module(self.val_sketches.to(pl_module.device))
        save_image(color_gen, str(self.save_dir / f'gen-{pl_module.current_epoch}.png'), normalize=True)
        trainer.logger.experiment.log({
            "examples": [wandb.Image(x) for x in color_gen]
        })
