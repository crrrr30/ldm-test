import tqdm
import logging

import torch
from torch.optim import Adam
import torch.nn.functional as F

# from convnext import *
from unet import Unet, base
from utils import *
from dataset import get_dataloader
from scheduler import ScheduledOptim

device = "cuda"
image_size = 64
batch_size = 128
total_steps = 30000
warmup_steps = 2000

model_config = base
max_timesteps = model_config["timesteps"]

logging.basicConfig(
    filename="logs.txt",
    filemode='w',
    format='%(asctime)s, %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG
)

dataloader = get_dataloader(batch_size=batch_size)

model = Unet(**model_config).to(device)
opt = Adam(model.parameters(), betas=(.9, .999))
sched = ScheduledOptim(opt, total_steps=total_steps, base=1e-3, decay_type="cosine", warmup_steps=warmup_steps)
count_parameters(model)

hyperparams = get_hyperparams(model_config["timesteps"], device)
pbar = tqdm.trange(total_steps + 1)

resume = None
if resume is not None:
    state = torch.load(resume, map_location=device)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    logging.info(f"All keys matched successfully, loaded from {resume}.")

for step in pbar:
    sched.zero_grad()
    data = next(dataloader).to(device)
    b = data.shape[0]
    if data.shape[-1] != image_size:
        data = F.interpolate(data, image_size, mode="bilinear", align_corners=True)
    t = torch.randint(0, max_timesteps, size=(b, )).to(device)
    loss = p_losses(hyperparams, model, data, t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    sched.step_and_update_lr()
    pbar.set_description(f'Loss: {loss.item():.6f}')
    logging.info(f"Step {step:010}, Loss: {loss.item():.6f}")
    if step % 1000 == 0:
        sample_and_save(hyperparams, model, step, image_size, num=8)
        torch.save({
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'model': model.state_dict(),
        }, f'./ckpt-{step:06}.pt')


# import matplotlib.pyplot as plt
# plt.imshow((data[0] + 1. / 2).clamp_(0., 1.).permute(1, 2, 0).detach().cpu().numpy()); plt.show()
# plt.imshow((corrupted_data[0] + 1. / 2).clamp_(0., 1.).permute(1, 2, 0).detach().cpu().numpy()); plt.show()
# plt.imshow((out[0] + 1. / 2).clamp_(0., 1.).permute(1, 2, 0).detach().cpu().numpy()); plt.show()
