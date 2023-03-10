import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, in_planes, planes, downsample = None):

        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.net = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, 1, 0, bias = False),
                    nn.InstanceNorm2d(planes),
                    nn.ReLU(),
                    nn.Conv2d(planes, planes, 3, 1, 1, bias = False),
                    nn.InstanceNorm2d(planes),
                    nn.ReLU(),
                    nn.Conv2d(planes, planes * self.expansion, 1, 1, 0, bias = False),
                    nn.InstanceNorm2d(planes * self.expansion)
                )
        self.relu = nn.ReLU()

        return

    def forward(self, x):

        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)

        y = self.relu(self.net(x) + residual)

        return y

        
class VoxResModule(nn.Module):

    def __init__(self, num_channels):

        super(VoxResModule, self).__init__()
        self.net = nn.Sequential(
                    nn.InstanceNorm2d(num_channels),
                    nn.ReLU(),
                    nn.Conv2d(num_channels, num_channels, 3, 1, 1),
                    nn.InstanceNorm2d(num_channels),
                    nn.ReLU(),
                    nn.Conv2d(num_channels, num_channels, 3, 1, 1)
                )

        return

    def forward(self, x):

        return x + self.net(x)

class VoxResNet(nn.Module):

    def __init__(self, in_channels, n_classes = 4, num_channels = 32):

        super(VoxResNet, self).__init__()
        self.stem_net = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, 2, 1, bias = False),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(),
                    self._make_layer(Bottleneck, 32, 16, 2)
                )
        self.net_h0 = nn.Sequential(
                    nn.Conv2d(64, num_channels, 3, 1, 1),
                    nn.InstanceNorm2d(num_channels),
                    nn.ReLU(),
                    nn.Conv2d(num_channels, num_channels, 3, 1, 1)
                )
        self.net_c0 = nn.Sequential(
                    nn.ConvTranspose2d(num_channels, num_channels, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(num_channels, n_classes, 3, 1, 1)
                )
        self.net_h1 = nn.Sequential(
                    nn.InstanceNorm2d(num_channels),
                    nn.ReLU(),
                    nn.Conv2d(num_channels, num_channels * 2, 3, 2, 1),
                    VoxResModule(num_channels = num_channels * 2),
                    VoxResModule(num_channels = num_channels * 2)
                )
        self.net_c1 = nn.Sequential(
                    nn.ConvTranspose2d(num_channels * 2, num_channels * 2, 4, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(num_channels * 2, n_classes, 3, 1, 1)
                )
        self.net_h2 = nn.Sequential(
                    nn.InstanceNorm2d(num_channels * 2),
                    nn.ReLU(),
                    nn.Conv2d(num_channels * 2, num_channels * 2, 3, 2, 1),
                    VoxResModule(num_channels = num_channels * 2),
                    VoxResModule(num_channels = num_channels * 2)
                )
        self.net_c2 = nn.Sequential(
                    nn.ConvTranspose2d(num_channels * 2, num_channels * 2, 6, 4, 1),
                    nn.ReLU(),
                    nn.Conv2d(num_channels * 2, n_classes, 3, 1, 1)
                )
        self.net_h3 = nn.Sequential(
                    nn.InstanceNorm2d(num_channels * 2),
                    nn.ReLU(),
                    nn.Conv2d(num_channels * 2, num_channels * 2, 3, 2, 1),
                    VoxResModule(num_channels = num_channels * 2),
                    VoxResModule(num_channels = num_channels * 2)
                )
        self.net_c3 = nn.Sequential(
                    nn.ConvTranspose2d(num_channels * 2, num_channels * 2, 10, 8, 1),
                    nn.ReLU(),
                    nn.Conv2d(num_channels * 2, n_classes, 3, 1, 1)
                )
        
        self.final_conv = nn.Sequential(
            nn.InstanceNorm2d(512),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.Conv2d(512, 512, 3, 1, 1, groups=512),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.ReLU()
        )

        return

    def _make_layer(self, block, in_planes, planes, num_block):

        if in_planes == planes * block.expansion:
            downsample = None
        else:
            downsample = nn.Sequential(
                        nn.Conv2d(in_planes, planes * block.expansion, 1, 1, 0, bias = False),
                        nn.InstanceNorm2d(planes * block.expansion)
                    )
        layers = [block(in_planes, planes, downsample)]
        layers.extend([block(planes * block.expansion, planes) for i in range(num_block - 1)])

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.stem_net(x)
        x = F.max_pool2d(x, 2)
        h = self.net_h0(x)
        h = F.max_pool2d(h, 2)
        c0 = self.net_c0(h)
        h = self.net_h1(h)
        h = F.max_pool2d(h, 2)
        c1 = self.net_c1(h)
        h = self.net_h2(h)
        h = F.max_pool2d(h, 2)
        c2 = self.net_c2(h)
        h = self.net_h3(h)
        h = F.max_pool2d(h, 2)
        c3 = self.net_c3(h)
        c0 = F.interpolate(c0, size = c3.shape[-2:], mode = 'bilinear', align_corners = False)
        c1 = F.interpolate(c1, size = c3.shape[-2:], mode = 'bilinear', align_corners = False)
        c2 = F.interpolate(c2, size = c3.shape[-2:], mode = 'bilinear', align_corners = False)
        c = c0 + c1 + c2 + c3

        c = c + self.final_conv(c)
 
        return c

if __name__ == "__main__":
    encoder = VoxResNet(3, 512)

    x = torch.randn(4, 3, 512, 512)
    y = encoder(x)
    print(y.shape)