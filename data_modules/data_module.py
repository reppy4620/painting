import pathlib
from hydra.utils import instantiate

import pytorch_lightning as pl
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_dir = pathlib.Path(cfg.data_dir)

        self.train_x = None
        self.valid_x = None

    def setup(self, stage=None):
        img_paths = list(self.data_dir.glob('**/*.*'))
        train_length = int(len(img_paths) * self.cfg.train_ratio)
        self.train_x = instantiate(self.cfg.dataset,
                                   img_paths=img_paths[:train_length],
                                   transform=self.train_transforms())
        self.valid_x = instantiate(self.cfg.dataset,
                                   img_paths=img_paths[train_length:],
                                   transform=self.val_transforms())

    def train_dataloader(self):
        return DataLoader(
            self.train_x,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_x,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

    def train_transforms(self):
        return nn.Sequential(
            T.RandomCrop((self.cfg.resolution, self.cfg.resolution)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        )

    def val_transforms(self):
        return nn.Sequential(
            T.RandomCrop((self.cfg.resolution, self.cfg.resolution))
        )
