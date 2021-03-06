import math
import torch.nn as nn

from ..common.layers import Concat
from ..common.utils import make_channels


class Generator(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        channels = make_channels(math.log2(resolution / 8))
        self.in_layer = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.downs = nn.ModuleList([
            self._build_down_layer(
                in_c,
                out_c
            ) for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        channels = channels[::-1]

        self.middle = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[0], channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.ups = nn.ModuleList([
            Concat(
                self._build_up_layer(
                    in_c*2,
                    out_c
                )
            ) for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.out_layer = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1]*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-1]*2),
            nn.GLU(dim=1),
            nn.Conv2d(channels[-1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    @staticmethod
    def _build_down_layer(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )

    @staticmethod
    def _build_up_layer(in_c, out_c):
        return nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_c, out_c*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c*2),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        x = self.in_layer(x)
        down_outputs = list()
        for down in self.downs:
            x = down(x)
            down_outputs.append(x)
        x = self.middle(x)
        down_outputs = down_outputs[::-1]
        for i, up in enumerate(self.ups):
            x = up(x, down_outputs[i])
        x = self.out_layer(x)
        return x
