from functools import partial

# import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import Permute
from torchvision.models.swin_transformer import SwinTransformerBlock

# from einops.layers.torch import Rearrange



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

        self.norm = norm_layer(dim)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.final_conv = nn.Conv2d(dim, input_dim, 1, 1, 0)

    def forward(self, x):
        x = self.init_conv(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.final_conv(x)
        return x

