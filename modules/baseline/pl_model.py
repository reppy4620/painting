import pytorch_lightning as pl
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief

from .discriminator import Discriminator
from .generator import Generator
from ..common.loss import BCEWithLogitsGANLoss


class BaselineModule(pl.LightningModule):
    def __init__(self, resolution):
        super().__init__()
        self.G = Generator(resolution)
        self.D = Discriminator(resolution)
        self.automatic_optimization = False

        self.criterion = BCEWithLogitsGANLoss()

    def forward(self, sketch):
        return self.G(sketch)

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt_g, opt_d = self.optimizers()

        sketch, color = batch

        # train G
        fake = self.G(sketch)
        pred_fake = self.D(fake)

        loss_g_gan = self.criterion.loss_g(pred_fake)
        recon_loss = F.mse_loss(fake, color)
        loss_g = loss_g_gan + 10 * recon_loss

        opt_g.zero_grad()
        self.manual_backward(loss_g, opt_g)
        opt_g.step()

        # train D
        fake = self.G(sketch)
        pred_fake = self.D(fake)
        pred_real = self.D(color)
        loss_d = self.criterion.loss_d(pred_fake, pred_real)

        opt_d.zero_grad()
        self.manual_backward(loss_d, opt_d)
        opt_d.step()

        self.log('loss_d', loss_d, prog_bar=True)
        self.log('loss_g', loss_g, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        sketch, color = batch

        fake = self.G(sketch)
        pred_fake = self.D(fake)

        loss_g_gan = self.criterion.loss_g(pred_fake)
        recon_loss = F.mse_loss(fake, color)
        loss_g = loss_g_gan + 10 * recon_loss

        fake = self.G(sketch)
        pred_fake = self.D(fake)
        pred_real = self.D(color)
        loss_d = self.criterion.loss_d(pred_fake, pred_real)

        val_loss = loss_g + loss_d

        self.log_dict({
            'val_loss': val_loss
        }, prog_bar=True)

    def configure_optimizers(self):
        opt_g = AdaBelief(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_d = AdaBelief(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))

        return [opt_g, opt_d]
