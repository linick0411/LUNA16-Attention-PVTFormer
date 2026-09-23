"""Original 2D PVTFormer baseline adapted to the shared project utilities."""

import torch
import torch.nn as nn

from model_utils import load_pvtv2_b3_weights
from pvtv2 import pvt_v2_b3


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=True, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x) if self.act else x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, inputs):
        return self.relu(self.conv(inputs) + self.shortcut(inputs))


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.residual = ResidualBlock(in_c + out_c, out_c)

    def forward(self, x, skip):
        return self.residual(torch.cat([self.up(x), skip], dim=1))


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.residual = ResidualBlock(in_c, out_c)

    def forward(self, inputs):
        return self.residual(self.up(inputs))


class PVTFormerBaseline(nn.Module):
    """Center-slice 2D baseline matching the upstream PVTFormer architecture."""

    def __init__(self):
        super().__init__()
        self.backbone = pvt_v2_b3()
        load_pvtv2_b3_weights(self.backbone)

        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)

        self.d1 = DecoderBlock(64, 64)
        self.d2 = DecoderBlock(64, 64)
        self.d3 = UpBlock(64, 64, 4)
        self.u1 = UpBlock(64, 64, 4)
        self.u2 = UpBlock(64, 64, 8)
        self.u3 = UpBlock(64, 64, 16)
        self.fuse = ResidualBlock(64 * 4, 64)
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(f"Baseline expects (B, 3, H, W), got {tuple(inputs.shape)}")

        e1, e2, e3 = self.backbone(inputs)[:3]
        c1, c2, c3 = self.c1(e1), self.c2(e2), self.c3(e3)
        d1 = self.d1(c3, c2)
        d2 = self.d2(d1, c1)
        d3 = self.d3(d2)
        features = torch.cat([d3, self.u1(c1), self.u2(c2), self.u3(c3)], dim=1)
        return self.output(self.fuse(features))
