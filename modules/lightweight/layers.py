import torch.nn as nn

from ..common.layers import BranchAdd


def DownLayer(in_c, out_c):
    return BranchAdd(
        nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        ),
        nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )
    )


def UpLayer(in_c, out_c):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(in_c, out_c*2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_c*2),
        nn.GLU(dim=1)
    )


class SLE(nn.Module):
    def __init__(self, low_c, high_c):
        super().__init__()
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(low_c, high_c, 4, bias=False),
            nn.SiLU(),
            nn.Conv2d(high_c, high_c, 1, bias=False)
        )

    def forward(self, low, high):
        return high * self.se_block(low)
