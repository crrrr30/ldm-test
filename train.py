from tqdm.notebook import tqdm

import torch
from torch.optim import AdamW
import torch.nn.functional as F

from torch.optim.lr_scheduler import LinearLR
import pytorch_lightning as pl

from diffusers.models import AutoencoderKL

from swin_decoder import SwinTransformerBackbone
from utils import *
from dataset import get_dataloader
from scheduler import ScheduledOptim


device = "cuda"
image_size = 256
latent_dim = 4
latent_size = image_size // 8
batch_size = 256
num_epochs = 120
half_precision = True


model_config = {
    "timesteps": 1000,
    "latent_dim": latent_dim,
    "latent_size": latent_size,
    "num_epochs": num_epochs
}
max_timesteps = model_config["timesteps"]

dataloader = get_dataloader(batch_size=batch_size)

class LitLatentDiffusion(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.hyperparams = get_hyperparams(model_config["timesteps"], device=self.device)
        self.num_epochs = model_config["num_epochs"]
        self.latent_dim = model_config["latent_dim"]
        self.latent_size = model_config["latent_size"]
        self.max_timesteps = model_config["max_timesteps"]
        self.model = SwinTransformerBackbone()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.vae.freeze()
        self.save_hyperparameters(ignore=["vae"])
    def forward(self, x):
        return self.model(x)
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-6, weight_decay=1e-2)
        scheduler = LinearLR(
            optimizer,
            start_factor=0.0015,
            end_factor=0.0195,
            total_iters=self.trainer.estimated_stepping_batches * self.num_epochs
        )
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "metric_to_track",
                },
            }
    def training_step(self, data, idx):
        sch = self.lr_schedulers()
        sch.zero_grad()
        sch.step()
        b = data.shape[0]
        if data.shape[-1] != self.image_size:
            data = F.interpolate(data, self.image_size, mode="bilinear", align_corners=True)
        t = torch.randint(0, max_timesteps, size=(b, ), device=self.device)
        with torch.no_grad():
            latents = self.vae.encoder(data).mode()
        loss = p_losses(self.hyperparams, self.model, latents, t)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    def on_save_checkpoint(self, checkpoint):
        print(checkpoint)
        # sample_and_save(self.hyperparams, self.model, self.vae, step, self.latent_size, num=8)
    def validation_step(self, data, idx):
        b = data.shape[0]
        with torch.no_grad():
            latents = self.vae.encode(data).mode()
        losses = []
        for t in range(self.max_timesteps):
            t = torch.Tensor([t] * b, device=self.device)
            loss = p_losses(self.hyperparams, self.model, latents, t)
            losses.append(loss)
        self.log("val_l1_loss", torch.mean(losses))


tb_logger = pl.loggers.TensorBoardLogger(
    "lightning_logs/",
)

model = LitLatentDiffusion(model_config)
trainer = pl.Trainer(
    default_root_dir=".",
    precision=16 if half_precision else 32,
    max_epochs=num_epochs,
    devices=torch.cuda.device_count(),
    accelerator="gpu",
    callbacks=[],
    logger=tb_logger
)

trainer.fit(model, dataloader)
