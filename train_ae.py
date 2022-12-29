import tqdm
import logging

import torch
from torch.optim import Adam
import torch.nn.functional as F

from myresdecoder import VoxResNet as Decoder
from myresencoder import VoxResNet as Encoder
from utils import *
from dataset import get_dataloader
from scheduler import ScheduledOptim

device = "cuda"
image_size = 512
batch_size = 32
total_steps = 30000
warmup_steps = 2000


logging.basicConfig(
    filename="logs.txt",
    filemode='w',
    format='%(asctime)s, %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG
)

dataloader = get_dataloader(batch_size=batch_size)

encoder = Encoder(3, 512).to(device)
decoder = Decoder(512, 3).to(device)
opt = Adam(list(encoder.parameters()) + list(decoder.parameters()), betas=(.9, .999))
sched = ScheduledOptim(opt, total_steps=total_steps, base=1e-3, decay_type="cosine", warmup_steps=warmup_steps)
count_parameters(encoder, model_name="Encoder")
count_parameters(decoder, model_name="Decoder")

pbar = tqdm.trange(total_steps + 1)

resume = None
if resume is not None:
    state = torch.load(resume, map_location=device)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    logging.info(f"All keys matched successfully, loaded from {resume}.")

epsilon = 1e-2

for step in pbar:
    sched.zero_grad()
    data = next(dataloader).to(device)
    b = data.shape[0]
    if data.shape[-1] != image_size:
        data = F.interpolate(data, image_size, mode="bilinear", align_corners=True)
    z = encoder(data + epsilon * torch.randn_like(data))
    reconstruction = decoder(z)
    loss = F.l1_loss(reconstruction, data)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.)
    sched.step_and_update_lr()
    pbar.set_description(f'Loss: {loss.item():.6f}')
    logging.info(f"Step {step:010}, Loss: {loss.item():.6f}")
    if step % 1000 == 0:
        torch.save({
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
        }, f'./ae-ckpt-{step:06}.pt')


# import matplotlib.pyplot as plt
# plt.imshow((data[0] + 1. / 2).clamp_(0., 1.).permute(1, 2, 0).detach().cpu().numpy()); plt.show()
# plt.imshow((corrupted_data[0] + 1. / 2).clamp_(0., 1.).permute(1, 2, 0).detach().cpu().numpy()); plt.show()
# plt.imshow((out[0] + 1. / 2).clamp_(0., 1.).permute(1, 2, 0).detach().cpu().numpy()); plt.show()
