import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange, Reduce


class SpatialGatingUnit(nn.Module):
    def __init__(self, n_tokens, dim):
        super().__init__()
        self.duplicate = nn.Linear(dim, dim * 2)
        self.ln = nn.LayerNorm(dim)
        self.dense = nn.Linear(n_tokens, n_tokens)

    def forward(self, x):
        x = self.duplicate(x)
        u, v = torch.chunk(x, 2, dim=-1)
        v = self.ln(v)
        v = self.dense(v.permute(0, 2, 1)).permute(0, 2, 1) + 1.      # Equiv. to bias init. = \vec 1
        return u * v


class gMLP(nn.Module):
    def __init__(self, n_tokens, d_model, d_ffn):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dense1 = nn.Linear(d_model, d_ffn)
        self.dense2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(n_tokens, d_ffn)

    def forward(self, x):
        shortcut = x
        x = self.ln(x)
        x = self.dense1(x)
        x = F.gelu(x)
        x = self.sgu(x)
        x = self.dense2(x)
        return shortcut + x


class Model(nn.Module):
    def __init__(self, image_size, patch_size=4, n_layers=6, d_model=256, d_ffn=1024):
        super().__init__()
        self.patch_size = patch_size
        n_tokens = image_size * image_size // patch_size // patch_size

        self.patchify = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size),
            nn.Linear(3 * patch_size * patch_size, d_model)
        )
        self.backbone = nn.ModuleList([
            gMLP(n_tokens=n_tokens, d_model=d_model, d_ffn=d_ffn)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            Reduce("b n d -> b d", reduction="mean"),
            nn.Linear(d_model, 10)
        )
    
    def forward(self, x):
        x = self.patchify(x)
        for layer in self.backbone:
            x = layer(x)
        return self.classifier(x)

model = Model(32)
print(f"#params: {sum([w.numel() for w in model.parameters()]):,}")
# print(model)
x = torch.randn(4, 3, 32, 32)
y = model(x)