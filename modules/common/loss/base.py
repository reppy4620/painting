import torch
import torch.nn as nn
from abc import abstractmethod, ABCMeta


class GANLossBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def loss_d(self, pred_fake, pred_real):
        pass

    @abstractmethod
    def loss_g(self, pred):
        pass


class NormalGANLossBase(GANLossBase):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def loss_d(self, pred_fake, pred_real):
        return self.criterion(pred_fake, torch.zeros_like(pred_fake)) + \
               self.criterion(pred_real, torch.ones_like(pred_real))

    def loss_g(self, pred):
        return self.criterion(pred, torch.ones_like(pred))
