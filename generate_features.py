from tqdm import tqdm
import logging

import torch
from torch.optim import Adam
import torch.nn.functional as F

from myresdecoder import VoxResNet as Decoder
from myresencoder import VoxResNet as Encoder
from utils import *
from dataset import get_dataloader
from scheduler import ScheduledOptim

if __name__ == "__main__":
    device = "cuda"
    image_size = 512
    batch_size = 256
    epsilon = 0
    N_MAX = 30000 * 4

    logging.basicConfig(
        filename="logs.txt",
        filemode='w',
        format='%(asctime)s, %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )

    dataloader = get_dataloader(batch_size=batch_size)

    encoder = Encoder(3, 512).to(device)

    pbar = tqdm(range(N_MAX))

    resume = "ae-ckpt-030000.pt"
    if resume is not None:
        state = torch.load(resume, map_location=device)
        encoder.load_state_dict(state["encoder"])
        logging.info(f"All keys matched successfully, loaded from {resume}.")

    encoder.eval()

    results = torch.empty(N_MAX, 512, 8, 8)

    with torch.no_grad():
        start = 0
        while start <= N_MAX:
            data = next(dataloader).to(device)
            b = data.shape[0]
            if data.shape[-1] != image_size:
                data = F.interpolate(data, image_size, mode="bilinear", align_corners=True)
            z = encoder(data)
            results[start : min(N_MAX, start + z.shape[0])] = z.cpu()[:min(N_MAX - start, z.shape[0])]
            start += z.shape[0]
            print(f"=> {start}")
            pbar.update(min(N_MAX - start, z.shape[0]))

    torch.save(results, "features.pt")