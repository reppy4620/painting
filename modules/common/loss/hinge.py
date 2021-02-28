from .base import GANLossBase
import torch.nn.functional as F


class HingeGANLoss(GANLossBase):

    def loss_d(self, pred_fake, pred_real):
        return (F.relu(1 + pred_fake) + F.relu(1 - pred_real)).mean()

    def loss_g(self, pred):
        return -pred.mean()
