import math
import random

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .layers import DownLayer, UpLayer
from ..common import make_channels


class SimpleDecoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layers = nn.Sequential(*[
            *[UpLayer(channel//(2**i), channel//(2**(i+1))) for i in range(4)],
            nn.Conv2d(channel//(2**4), 3, 3, 1, 1, bias=False),
            nn.Tanh()
        ])

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        assert resolution in [256, 512]
        self.crop_size = 128
        self.resize = T.Resize((self.crop_size, self.crop_size))

        channels = make_channels(int(math.log2(resolution / 8)))
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.downs = nn.ModuleList([
            DownLayer(
                in_c,
                out_c
            ) for in_c, out_c in zip(channels[:-1], channels[1:])
        ])
        self.out_layer = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[-1], 1, 4, bias=False)
        )
        self.I_decoder = SimpleDecoder(channels[-1])
        self.I_part_decoder = SimpleDecoder(channels[-2])

    @staticmethod
    def half_from_part_idx(x, part_idx):
        h, w = x.size()[2:]
        half_h, half_w = h // 2, w // 2
        if part_idx == 0:
            x_part = x[:, :, :half_h, :half_w]
        elif part_idx == 1:
            x_part = x[:, :, :half_h, half_w:]
        elif part_idx == 2:
            x_part = x[:, :, half_h:, :half_w]
        elif part_idx == 3:
            x_part = x[:, :, half_h:, half_w:]
        else:
            raise ValueError()
        return x_part

    def prepare_dec_image(self, color):
        I = self.resize(color)
        part_idx = random.randint(0, 3)
        I_part = self.resize(self.half_from_part_idx(color, part_idx))
        return I, I_part, part_idx

    def forward(self, x, with_recon=False):
        if with_recon:
            color = x
        x = self.in_layer(x)
        down_outputs = list()
        for layer in self.downs:
            x = layer(x)
            down_outputs.append(x)
        x = self.out_layer(x)
        if not with_recon:
            return x

        I, I_part, part_idx = self.prepare_dec_image(color)
        dec_inp_I = down_outputs[-1]
        dec_inp_I_part = self.half_from_part_idx(down_outputs[-2], part_idx)
        I_hat = self.I_decoder(dec_inp_I)
        I_part_hat = self.I_part_decoder(dec_inp_I_part)

        recon_loss = F.mse_loss(I, I_hat) + F.mse_loss(I_part, I_part_hat)

        return x, recon_loss
