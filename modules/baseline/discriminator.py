import math
import torch.nn as nn

from ..common import make_channels


class Discriminator(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        channels = make_channels(math.log2(resolution / 8))
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layers = nn.Sequential(*[
            self._build_layer(
                in_c,
                out_c
            ) for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.out_layer = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 4)
        )

    def _build_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.layers(x)
        x = self.out_layer(x)
        return x
