import torch
import torch.nn as nn
import torch.nn.functional as F


def mish(x):
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    def forward(self, x):
        return mish(x)


def tanhexp(x):
    return x * torch.tanh(torch.exp(x))


class TanhExp(nn.Module):
    def forward(self, x):
        return tanhexp(x)
