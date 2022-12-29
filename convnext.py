from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, time_embed_dim=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if time_embed_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.GELU(),
                nn.Linear(time_embed_dim, dim)
            )

    def forward(self, x, t=None):
        if t is not None:
            x = x + rearrange(self.time_mlp(t), 'b c -> b c 1 1')
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x      # Channel-wise scalar multiple
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 time_embed_dim=32, timesteps=None
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value,
                time_embed_dim=time_embed_dim) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.final_conv = nn.Sequential(
            Block(sum(dims), time_embed_dim=time_embed_dim),
            nn.Conv2d(sum(dims), in_chans, 1, 1, 0),
        )

        if time_embed_dim is not None:
            self.time_embedding = nn.Embedding(timesteps, time_embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x, t):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x, t=None):
        if t is not None:
            t = self.time_embedding(t)
        shape = x.shape[-2:]
        x = self.forward_features(x, t)
        rescaled_features = torch.concat(
            [F.interpolate(feature, shape, mode="bilinear", align_corners=True) for feature in x],
            axis=1
        )   
        return self.final_conv(rescaled_features)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


tiny = dict(
    in_chans=3,
    depths=[3, 3, 9, 3], 
    dims=[96, 192, 384, 768], 
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    out_indices=[0, 1, 2, 3],
    timesteps=1000
)

base = dict(
    in_chans=3,
    depths=[3, 3, 27, 3], 
    dims=[128, 256, 512, 1024], 
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    out_indices=[0, 1, 2, 3],
    timesteps=1000
)

large = dict(
    in_chans=3,
    depths=[3, 3, 27, 3], 
    dims=[192, 384, 768, 1536], 
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    out_indices=[0, 1, 2, 3],
    timesteps=1000
)

xlarge = dict(
    in_chans=3,
    depths=[3, 3, 27, 3], 
    dims=[256, 512, 1024, 2048], 
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    out_indices=[0, 1, 2, 3],
    timesteps=1000
)


# Simple Conv


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, time_embed_dim=None):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, 3, 1, 1)
        self.res_conv = None if in_chans == out_chans else nn.Conv2d(in_chans, out_chans, 1, 1, 0)
        if time_embed_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.GELU(),
                nn.Linear(time_embed_dim, in_chans)
            )
    def forward(self, x, t=None):
        if t is not None:
            x = x + rearrange(self.time_mlp(t), "b c -> b c 1 1")
        y = self.norm(x)
        y = self.conv1(y)
        y = F.gelu(y)
        y = self.conv2(y)
        return x + y if self.res_conv is None else self.res_conv(x) + y

class ConditionedSequential(nn.Module):
    def __init__(self, sub_modules):
        super().__init__()
        self.sub_modules = sub_modules
    def forward(self, x, t):
        for f in self.sub_modules:
            x = f(x, t)
        return x

class ConvNet(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], time_embed_dim=32, timesteps=None):
        super().__init__()
        total_dims = [in_chans] + dims
        self.init_convs = nn.ModuleList([
            nn.Conv2d(total_dims[i], total_dims[i + 1], 1, 1, 0)
            for i in range(len(dims))
        ])
        self.stages = nn.ModuleList([
            ConditionedSequential(nn.ModuleList([ConvBlock(dim, dim, time_embed_dim=time_embed_dim) for _ in range(depth)]))
            for dim, depth in zip(dims, depths)
        ])
        self.final_conv = ConvBlock(sum(dims), 3, time_embed_dim=time_embed_dim)
        if time_embed_dim is not None:
            self.time_embedding = nn.Embedding(timesteps, time_embed_dim)
    def forward(self, x, t=None):
        if t is not None:
            t = self.time_embedding(t)
        features = []
        init_size = min(x.shape[2], x.shape[3])      # min{H, W}
        size = init_size
        for init_conv, stage in zip(self.init_convs, self.stages):
            x = init_conv(x)
            x = stage(x, t)
            size = size // 2
            x = F.interpolate(x, size, mode="bilinear", align_corners=True)
            features.append(x)
        features = torch.concat([
            F.interpolate(feature, init_size, mode="bilinear", align_corners=True)
            for feature in features
        ], axis=1)
        return self.final_conv(features, t)


tiny = dict(
    in_chans=3,
    depths=[3, 3, 9, 3], 
    dims=[96, 192, 384, 768], 
    time_embed_dim=32,
    timesteps=1000
)