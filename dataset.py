from PIL import Image
from pathlib import Path
from functools import partial
from multiprocessing import cpu_count

import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


def cycle(dl):
    while True:
        for data in dl:
            yield data

class CustomDataset(Dataset):
    def __init__(
        self,
        image_size=256,
        augment_horizontal_flip=True,
        convert_image_to=None
    ):
        super().__init__()
        self.image_size = image_size

        maybe_convert_fn = partial(convert_image_to, convert_image_to) if convert_image_to is not None else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(.5, .5)
        ])

    def __len__(self):
        return 30_000

    def __getitem__(self, index):
        img = Image.open(f"./celeba_hq_256/{index:05}.jpg")
        return self.transform(img)


def get_dataloader(batch_size):
    dataset = CustomDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
    dataloader = cycle(dataloader)
    return dataloader