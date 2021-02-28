import pathlib

import pytorch_lightning as pl
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader

from .get_dataset import get_dataset
from .utils import OneOf


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.dataset_type = cfg.dataset
        self.data_dir = pathlib.Path(cfg.data_dir)
        self.resolution = cfg.resolution
        self.train_ratio = cfg.train_ratio
        self.batch_size = cfg.batch_size

        self.train_x = None
        self.valid_x = None

    def setup(self, stage=None):
        Dataset = get_dataset(self.dataset_type)
        img_paths = list(self.data_dir.glob('**/*.*'))
        train_length = int(len(img_paths) * self.train_ratio)
        self.train_x = Dataset(img_paths[:train_length], self.train_transforms())
        self.valid_x = Dataset(img_paths[train_length:], self.val_transforms())

    def train_dataloader(self):
        return DataLoader(
            self.train_x,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_x,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8
        )

    def train_transforms(self):
        return nn.Sequential(
            OneOf(
                T.RandomResizedCrop((self.resolution, self.resolution)),
                T.Resize((self.resolution, self.resolution))
            ),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5)], p=0.1),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        )

    def val_transforms(self):
        return nn.Sequential(
            T.CenterCrop((self.resolution, self.resolution))
        )
