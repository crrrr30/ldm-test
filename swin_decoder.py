from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import Permute
from torchvision.models.swin_transformer import SwinTransformerBlock

from einops.layers.torch import Rearrange


# Channel last!!
def _patch_expanding_rearrangement(x: torch.Tensor):
    H, W, C = x.shape[-3:]
    assert C % 4 == 0
    c = C // 4
    x0 = x[..., :c]; x1 = x[..., c : 2 * c]; x2 = x[..., 2 * c : 3 * c]; x3 = x[..., 3 * c : 4 * c];
    x = torch.empty(*x.shape[:-3], 2 * H, 2 * W, c)
    x[..., 0::2, 0::2, :] = x0
    x[..., 1::2, 0::2, :] = x1
    x[..., 0::2, 1::2, :] = x2
    x[..., 1::2, 1::2, :] = x3
    return x


class PatchExpanding(nn.Module):
    """Patch Expanding Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., 2*H, 2*W, C/2]
        """
        x = self.reduction(x)  # ... H W 2*C
        x = self.norm(x)
        x = _patch_expanding_rearrangement(x)       # ... H W 2*C -> ... 2*H 2*W C/2
        return x


class SwinTransformerBackbone(nn.Module):
    def __init__(
        self,
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
        block = None,
        upsample_layer = PatchExpanding,
    ):
        super().__init__()

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.patchify = nn.Sequential(
            Permute([0, 2, 3, 1]),
            norm_layer(embed_dim),
        )

        layers = []

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage = []
            dim = embed_dim // 2 ** (i_stage + 1)

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
            layers.append(nn.Sequential(upsample_layer(dim * 2, norm_layer), *stage))
        self.features = nn.ModuleList(layers)

        self.norm = norm_layer(dim)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.depatchify = nn.Sequential(
            Rearrange("b (c p1 p2) h w -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size),
            nn.Conv2d(dim // patch_size ** 2, 3, 1, 1, 0)
        )

    def forward(self, x):
        x = self.patchify(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.depatchify(x)
        return x


bb = SwinTransformerBackbone(
    embed_dim=512,
    depths=[4, 4, 8, 4],
    num_heads=[16, 16, 16, 16],
    window_size=[7, 7],
    stochastic_depth_prob=0.2,
)

x = torch.randn(2, 512, 8, 8)
y = bb(x)

print(f"#params: {sum([w.numel() for w in bb.parameters()]):,}")