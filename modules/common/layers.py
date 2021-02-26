import torch
import torch.nn as nn
import torch.nn.functional as F


class Concat(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x_down, x_up):
        x_up = F.interpolate(x_up, size=x_down.size()[2:])
        x = self.layer(torch.cat([x_down, x_up], dim=1))
        return x


class Branch(nn.Module):
    def __init__(self, layer1, layer2, op):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.op = op

    def forward(self, x):
        return self.op(self.layer1(x), self.layer2(x))


def BranchAdd(layer1, layer2):
    return Branch(layer1, layer2, op=lambda x, y: x + y)


class Identity(nn.Module):
    def forward(self, x):
        return x
