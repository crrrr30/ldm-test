import torch
from torch.optim import AdamW
import torch.nn.functional as F

from torch.optim.lr_scheduler import LinearLR
import pytorch_lightning as pl

from diffusers.models import AutoencoderKL

from swin_decoder import SwinTransformerBackbone
from utils import *
from dataset import get_dataloader

def to_device(hyperparams: dict, device):
    for key in hyperparams.keys():
        if torch.is_tensor(hyperparams[key]):
            hyperparams[key] = hyperparams[key].to(device)
    return hyperparams

class LitLatentDiffusion(pl.LightningModule):
        def __init__(self, model_config):
            super().__init__()
            self.automatic_optimization = False
            self.hyperparams = get_hyperparams(model_config["timesteps"], device=self.device)
            self.num_epochs = model_config["num_epochs"]
            self.latent_dim = model_config["latent_dim"]
            self.latent_size = model_config["latent_size"]
            self.max_timesteps = model_config["timesteps"]
            self.model = SwinTransformerBackbone(
                input_dim=4,
                embed_dim=256,
                depths=[4, 4, 8, 4],
                num_heads=[16, 16, 16, 16],
                window_size=[3, 3],
                stochastic_depth_prob=0.2,
            )
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
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
            return [optimizer], [scheduler]
        def training_step(self, data, idx):
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.zero_grad()
            b = data.shape[0]
            # if data.shape[-1] != self.image_size:
            #     data = F.interpolate(data, self.image_size, mode="bilinear", align_corners=True)
            t = torch.randint(0, self.max_timesteps, size=(b, ), device=self.device)
            with torch.no_grad():
                latents = self.vae.encode(data).latent_dist.sample() * 0.18215
            loss = p_losses(to_device(self.hyperparams, self.device), self.model, latents, t)
            self.log("train_loss", loss, prog_bar=True)
            self.manual_backward(loss)
            optimizer.step()
            scheduler.step()
            return loss
        def on_save_checkpoint(self, checkpoint):
            if self.current_epoch % 10 == 0:
                sample_and_save(self.hyperparams, self.model, self.vae, self.current_epoch, self.latent_size, num=16)
        # def validation_step(self, data, idx):
        #     b = data.shape[0]
        #     with torch.no_grad():
        #         latents = self.vae.encode(data).latent_dist.sample() * 0.18215
        #     losses = []
        #     for t in range(self.max_timesteps):
        #         t = torch.Tensor([t] * b, device=self.device)
        #         loss = p_losses(to_device(self.hyperparams, self.device), self.model, latents, t)
        #         losses.append(loss)
        #     self.log("val_l1_loss", torch.mean(losses))

def main():
    image_size = 256
    latent_dim = 4
    latent_size = image_size // 8
    batch_size = 128
    num_epochs = 850
    half_precision = True


    model_config = {
        "timesteps": 1000,
        "latent_dim": latent_dim,
        "latent_size": latent_size,
        "num_epochs": num_epochs
    }

    dataloader = get_dataloader(batch_size=batch_size)


    tb_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/",
    )

    # model = LitLatentDiffusion(model_config)
    model = LitLatentDiffusion.load_from_checkpoint("lightning_logs/lightning_logs/version_5/checkpoints/epoch=150-step=17818.ckpt")
    model.vae.requires_grad_(False)
    trainer = pl.Trainer(
        default_root_dir=".",
        precision=16 if half_precision else 32,
        max_epochs=num_epochs,
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        callbacks=[],
        logger=tb_logger,
    )

    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()