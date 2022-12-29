import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt

from unet import *
from utils import *

device = "cpu"
image_size = 64
num_images = 4
checkpoint = "ckpt-030000.pt"

with torch.no_grad():
    model = Unet(**base).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device)["model"]
    )
    hyperparams = get_hyperparams(base["timesteps"], device)
    img = p_sample_loop_final(hyperparams, model, shape=[4, 3, image_size, image_size])
    
    for i in range(num_images):
        img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
    save_image(torch.Tensor(img), f"sample-{checkpoint.split('.')[0]}.png")