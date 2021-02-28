import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from adabelief_pytorch import AdaBelief

from .generator import Generator
from .discriminator import Discriminator


class BaselineModule(pl.LightningModule):
    def __init__(self, resolution):
        super().__init__()
        self.G = Generator(resolution)
        self.D = Discriminator(resolution)
        self.automatic_optimization = False

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sketch):
        return self.G(sketch)

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt_g, opt_d = self.optimizers()

        sketch, color = batch

        # train G
        fake = self.G(sketch)
        pred_fake = self.D(fake)

        loss_g_gan = self.criterion(pred_fake, torch.ones_like(pred_fake))
        recon_loss = F.mse_loss(fake, color)
        loss_g = loss_g_gan + 10 * recon_loss

        opt_g.zero_grad()
        self.manual_backward(loss_g, opt_g)
        opt_g.step()

        # train D
        fake = self.G(sketch)
        pred_fake = self.D(fake)
        pred_real = self.D(color)
        loss_d = self.criterion(pred_fake, torch.zeros_like(pred_fake)) + \
                 self.criterion(pred_real, torch.ones_like(pred_real))

        opt_d.zero_grad()
        self.manual_backward(loss_d, opt_d)
        opt_d.step()

        self.log('loss_d', loss_d, prog_bar=True)
        self.log('loss_g', loss_g, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        sketch, color = batch

        fake = self.G(sketch)
        pred_fake = self.D(fake)

        loss_g_gan = -pred_fake.mean()
        recon_loss = F.mse_loss(fake, color)
        loss_g = loss_g_gan + 10 * recon_loss

        fake = self.G(sketch)
        pred_fake = self.D(fake)
        pred_real = self.D(color)
        loss_d = self.criterion(pred_fake, torch.zeros_like(pred_fake)) + \
                 self.criterion(pred_real, torch.ones_like(pred_real))

        val_loss = loss_g + loss_d

        self.log_dict({
            'val_loss': val_loss
        }, prog_bar=True)

    def configure_optimizers(self):
        opt_g = AdaBelief(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_d = AdaBelief(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))

        return [opt_g, opt_d]
