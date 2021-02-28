import math
import torch.nn as nn

from .layers import DownLayer, UpLayer, SLE
from ..common import make_channels, Concat


class Generator(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        channels = make_channels(int(math.log2(resolution / 8)))
        self.in_layer = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.downs = nn.ModuleList([
            DownLayer(
                in_c,
                out_c
            ) for in_c, out_c in zip(channels[:-1], channels[1:])
        ])
        channels = channels[::-1]
        self.ups = nn.ModuleList([
            UpLayer(in_c, out_c) if i == 0 else Concat(UpLayer(in_c * 2, out_c))
            for i, (in_c, out_c) in enumerate(zip(channels[:-1], channels[1:]))
        ])
        self.n_sles = (len(channels)-1) // 2
        self.sles = nn.ModuleList([
            SLE(
                low_c,
                high_c
            ) for low_c, high_c in zip(channels[1:self.n_sles + 1], channels[-self.n_sles:])
        ])
        self.out_layer = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1]*2, 1, bias=False),
            nn.BatchNorm2d(channels[-1]*2),
            nn.GLU(dim=1),
            nn.Conv2d(channels[-1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.in_layer(x)
        # down
        down_outputs = list()
        for layer in self.downs:
            x = layer(x)
            down_outputs.append(x)
        # up [:n_sles]
        down_outputs = down_outputs[::-1]
        up_outputs = list()
        for i, layer in enumerate(self.ups[:self.n_sles]):
            x = layer(x) if i == 0 else layer(x, down_outputs[i])
            up_outputs.append(x)
        # up [n_sles:-n_sles]
        s = slice(self.n_sles, -self.n_sles)
        for layer, mem in zip(self.ups[s], down_outputs[s]):
            x = layer(x, mem)
        # up [-n_sles:]
        for layer, sle, down_out, up_out in zip(self.ups[-self.n_sles:], self.sles,
                                                down_outputs[-self.n_sles:], up_outputs):
            x = layer(down_out, x)
            x = sle(up_out, x)
        x = self.out_layer(x)
        return x
