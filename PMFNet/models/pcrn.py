"""
Main model architecture combining all components
"""
import torch
import torch.nn as nn
from .blocks import ChLayer, GAB, PASPP
from .decoder import Decoder


class Residual(nn.Module):
    """Residual block wrapper"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PCRN(nn.Module):
    """Polyp Segmentation Network with Channel Layers, GAB modules, and Decoders"""
    def __init__(self, backbone, channel_index=[64, 128, 320, 512]):
        super(PCRN, self).__init__()
        
        # Parent Encoder (PVT V2 backbone)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]
        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        # Children Encoder - Channel Layers
        self.ChLayer1 = ChLayer(channel_index[0])
        self.ChLayer2 = ChLayer(channel_index[1])
        self.ChLayer3 = ChLayer(channel_index[2])
        self.ChLayer4 = ChLayer(channel_index[3])
        
        # Projection convolutions
        self.conv1 = nn.Conv2d(channel_index[0], channel_index[1], kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(channel_index[1], channel_index[2], kernel_size=1, padding='same')
        self.conv3 = nn.Conv2d(channel_index[2], channel_index[3], kernel_size=1, padding='same')
        self.MP = nn.MaxPool2d(kernel_size=2, stride=2)

        # Attention Modules - Group Aggregation Bridge
        self.GAB1 = GAB(channel_index[1], channel_index[0])
        self.GAB2 = GAB(channel_index[2], channel_index[1])
        self.GAB3 = GAB(channel_index[3], channel_index[2])

        # Decoder Module
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder3 = Decoder(channel_index[3], channel_index[2])
        self.decoder2 = Decoder(channel_index[2], channel_index[1])
        self.decoder1 = Decoder(channel_index[1], channel_index[0])

        # Output layer
        self.conv_out = nn.Conv2d(channel_index[0], 1, kernel_size=1, padding='same')
        self.up_out = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

        # Bottleneck - PASPP module
        self.b5 = PASPP(channel_index[3], channel_index[3])

    def get_pyramid(self, x):
        """Extract multi-scale features from backbone"""
        pyramid = []
        B = x.shape[0]
        
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)
        
        return pyramid

    def forward(self, x):
        # Parent Encoder - get multi-scale features from PVT V2
        pyramid = self.get_pyramid(x)
        p4 = pyramid[3]  # Deepest level
        p3 = pyramid[2]
        p2 = pyramid[1]
        p1 = pyramid[0]  # Shallowest level

        # Children Encoder - apply channel attention
        c1 = self.ChLayer1(p1)
        d1 = c1
        c1 = self.conv1(self.MP(c1))
        c1 = c1 + p2
        
        c2 = self.ChLayer2(c1)
        d2 = c2
        c2 = self.conv2(self.MP(c2))
        c2 = c2 + p3
        
        c3 = self.ChLayer3(c2)
        d3 = c3
        c3 = self.conv3(self.MP(c3))
        c3 = c3 + p4
        
        c4 = self.ChLayer4(c3)
        d4 = c4

        # Bottleneck processing
        bottleneck = self.b5(d4)

        # Decoder with skip connections via GAB
        de4, mask4 = self.decoder3(bottleneck)
        bfeb3 = self.GAB3(self.up(d4), d3, mask4)
        decoder3 = bfeb3 + de4

        de3, mask3 = self.decoder2(decoder3)
        bfeb2 = self.GAB2(self.up(bfeb3), d2, mask3)
        decoder2 = bfeb2 + de3

        de2, mask2 = self.decoder1(decoder2)
        bfeb1 = self.GAB1(self.up(bfeb2), d1, mask2)
        decoder1 = bfeb1 + de2

        # Final prediction
        out = self.conv_out(self.up_out(decoder1))
        out = self.sigmoid(out)

        return out
