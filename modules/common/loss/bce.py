import torch.nn as nn
from .base import NormalGANLossBase


def BCEGANLoss():
    return NormalGANLossBase(nn.BCELoss())


def BCEWithLogitsGANLoss():
    return NormalGANLossBase(nn.BCEWithLogitsLoss())
