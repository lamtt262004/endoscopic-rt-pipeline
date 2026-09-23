"""
Decoder module for segmentation
"""
import torch
import torch.nn as nn


class up_conv(nn.Module):
    """Upsampling and Convolution"""
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Decoder(nn.Module):
    """Decoder with boundary attention mask generation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = up_conv(in_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv11 = nn.Conv2d(out_channels, 1, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.up(x)
        decoder = x
        x = self.conv11(x)
        pred = self.sigmoid(x)
        
        # Compute Boundary Attention
        B_Att = 1 - torch.abs(pred - 0.5) / 0.5
        
        # Compute Foreground Attention
        F_Att = torch.abs(0.5 - pred) - B_Att
        Fi = F_Att - 0.5 * B_Att
        
        return decoder, Fi
