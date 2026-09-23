"""
Attention blocks and feature enhancement modules
Includes: ChLayer, BFEB, CBR, GAB, PASPP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChLayer(nn.Module):
    """Channel Attention Layer"""
    def __init__(self, in_channels):
        super(ChLayer, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding='same')
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.AAP = nn.AdaptiveAvgPool2d(1)
        self.cwc_down = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.cwc_up = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.BN(x)
        x = self.relu(x)
        x1 = x
        x = self.AAP(x)
        x = self.cwc_down(x)
        x = self.relu(x)
        x = self.cwc_up(x)
        x = self.sigmoid(x)
        x = x * x1
        return x


class BasicConv(nn.Module):
    """Basic Convolutional Block"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, 
                             dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.GELU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    """Flatten layer"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    """Channel Attention Gate"""
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    """Log-sum-exp pooling"""
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    """Channel Pooling"""
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    """Spatial Attention Gate"""
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class BFEB(nn.Module):
    """Boundary Feature Enhancement Block"""
    def __init__(self):
        super(BFEB, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map, pred):
        """
        Args:
            feature_map: Feature tensor (B, C, H, W)
            pred: Prediction tensor before sigmoid (B, 1, H, W)
        """
        pred = self.sigmoid(pred)
        
        # Boundary Attention
        B_Att = 1 - torch.abs(pred - 0.5) / 0.5
        
        # Foreground Attention
        F_Att = torch.abs(0.5 - pred) - B_Att
        
        Fi = F_Att - 0.5 * B_Att
        out = feature_map * Fi
        output = out + feature_map
        
        return output


class CBR(nn.Module):
    """Conv-BatchNorm-ReLU Block"""
    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.relu(x)
        return x


class LayerNorm(nn.Module):
    """Custom LayerNorm as used in GAB"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GAB(nn.Module):
    """Group Aggregation Bridge Module"""
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 3, 4, 5, 6]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl

        # Group 1: Multi-scale dilated convolution
        self.g1_1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding='same', dilation=d_list[0], groups=group_size + 1)
        )
        self.g1_2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding='same', dilation=d_list[2], groups=group_size + 1)
        )
        self.g1_3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding='same', dilation=d_list[4], groups=group_size + 1)
        )

        # Group 2: Multi-scale dilated convolution
        self.g2_1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding='same', dilation=d_list[1], groups=group_size + 1)
        )
        self.g2_2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding='same', dilation=d_list[3], groups=group_size + 1)
        )
        self.g2_3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding='same', dilation=d_list[5], groups=group_size + 1)
        )

        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 2, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 2, dim_xl, 1)
        )
        self.conv_out = nn.Conv2d(dim_xl * 3 + 3, dim_xl + 1, kernel_size=1, padding='same')

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)

        # Split into 2 groups
        xh = torch.chunk(xh, 2, dim=1)
        xl = torch.chunk(xl, 2, dim=1)

        # Group 1
        x1_1 = self.g1_1(torch.cat((xh[0], xl[0], mask), dim=1))
        x1_2 = self.g1_2(torch.cat((xh[0], xl[0], mask), dim=1))
        x1_3 = self.g1_3(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.conv_out(torch.cat((x1_1, x1_2, x1_3), dim=1))

        # Group 2
        x2_1 = self.g2_1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2_2 = self.g2_2(torch.cat((xh[1], xl[1], mask), dim=1))
        x2_3 = self.g2_3(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.conv_out(torch.cat((x2_1, x2_2, x2_3), dim=1))

        # Concatenate results from both groups
        x = torch.cat((x1, x2), dim=1)
        x = self.tail_conv(x)
        return x


class PASPP(nn.Module):
    """Pyramid Atrous Spatial Pyramid Pooling"""
    def __init__(self, inplanes, outplanes, output_stride=4, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        if output_stride == 4:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 4, 6, 10]
        elif output_stride == 2:
            dilations = [1, 12, 24, 36]
        elif output_stride == 16:
            dilations = [1, 2, 3, 4]
        elif output_stride == 1:
            dilations = [1, 16, 32, 48]
        else:
            raise NotImplementedError

        self._norm_layer = BatchNorm
        self.silu = nn.SiLU(inplace=True)
        self.conv1 = self._make_layer(inplanes, inplanes // 4)
        self.conv2 = self._make_layer(inplanes, inplanes // 4)
        self.conv3 = self._make_layer(inplanes, inplanes // 4)
        self.conv4 = self._make_layer(inplanes, inplanes // 4)
        self.atrous_conv1 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[0], padding=dilations[0])
        self.atrous_conv2 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[1], padding=dilations[1])
        self.atrous_conv3 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[2], padding=dilations[2])
        self.atrous_conv4 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[3], padding=dilations[3])
        self.conv5 = self._make_layer(inplanes // 2, inplanes // 2)
        self.conv6 = self._make_layer(inplanes // 2, inplanes // 2)
        self.convout = self._make_layer(inplanes, inplanes)

    def _make_layer(self, inplanes, outplanes):
        layer = []
        layer.append(nn.Conv2d(inplanes, outplanes, kernel_size=1))
        layer.append(nn.BatchNorm2d(outplanes))
        layer.append(self.silu)
        return nn.Sequential(*layer)

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.conv2(X)
        x3 = self.conv3(X)
        x4 = self.conv4(X)

        x12 = torch.add(x1, x2)
        x34 = torch.add(x3, x4)

        x1 = torch.add(self.atrous_conv1(x1), x12)
        x2 = torch.add(self.atrous_conv2(x2), x12)
        x3 = torch.add(self.atrous_conv3(x3), x34)
        x4 = torch.add(self.atrous_conv4(x4), x34)

        x12 = torch.cat([x1, x2], dim=1)
        x34 = torch.cat([x3, x4], dim=1)

        x12 = self.conv5(x12)
        x34 = self.conv5(x34)
        x = torch.cat([x12, x34], dim=1)
        x = self.convout(x)
        return x
