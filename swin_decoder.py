import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import Permute
from torchvision.models.swin_transformer import SwinTransformerBlock

# from einops.layers.torch import Rearrange

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SwinTransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        depths,
        num_heads,
        window_size,
        patch_size=4,
        mlp_ratio = 4.0,
        dropout = 0.0,
        attention_dropout = 0.0,
        stochastic_depth_prob = 0.1,
        norm_layer = None,
        block = None
    ):
        super().__init__()

        self.time = TimestepEmbedder(embed_dim)

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.init_conv = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, 1, 1, 0),
            Permute([0, 2, 3, 1]),
            norm_layer(embed_dim)
        )

        layers = []

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage = []
            dim = embed_dim

            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            stage = nn.Sequential(*stage)
            layers.append(stage)
        self.features = nn.ModuleList(layers)
        self.time_mapping = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 4),
                nn.SiLU(),
                nn.Linear(embed_dim // 4, embed_dim)
            )
        for _ in range(len(self.features))])

        self.norm = norm_layer(dim)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.final_conv = nn.Conv2d(dim, input_dim, 1, 1, 0)

    def forward(self, x, t):
        t = self.time(t)
        x = self.init_conv(x)
        for time_mapping, layer in zip(self.time_mapping, self.features):
            x = layer(x)
            x += time_mapping(t).view(t.shape[0], 1, 1, -1)
        x = self.norm(x)
        x = self.permute(x)
        x = self.final_conv(x)
        return x

