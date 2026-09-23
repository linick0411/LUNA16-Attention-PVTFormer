"""2.5D concatenation control without an added attention mechanism."""

import torch
import torch.nn as nn

from model_baseline import Conv2D, DecoderBlock, ResidualBlock, UpBlock
from model_utils import load_pvtv2_b3_weights
from pvtv2 import pvt_v2_b3


class PVTFormerConcat25D(nn.Module):
    """Fuse three neighboring slices with 1x1 convolutions only."""

    def __init__(self):
        super().__init__()
        self.backbone = pvt_v2_b3()
        load_pvtv2_b3_weights(self.backbone)

        self.fuse1 = nn.Conv2d(64 * 3, 64, kernel_size=1)
        self.fuse2 = nn.Conv2d(128 * 3, 128, kernel_size=1)
        self.fuse3 = nn.Conv2d(320 * 3, 320, kernel_size=1)
        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)
        self.d1 = DecoderBlock(64, 64)
        self.d2 = DecoderBlock(64, 64)
        self.d3 = UpBlock(64, 64, 4)
        self.u1 = UpBlock(64, 64, 4)
        self.u2 = UpBlock(64, 64, 8)
        self.u3 = UpBlock(64, 64, 16)
        self.fuse_output = ResidualBlock(64 * 4, 64)
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, inputs):
        if inputs.ndim != 5 or inputs.shape[1] != 3:
            raise ValueError(f"2.5D control expects (B, 3, C, H, W), got {tuple(inputs.shape)}")

        batch, slices, channels, height, width = inputs.shape
        if channels == 1:
            inputs = inputs.repeat(1, 1, 3, 1, 1)
        elif channels != 3:
            raise ValueError(f"Each slice must have 1 or 3 channels, got {channels}")
        inputs = inputs.reshape(batch * slices, 3, height, width)

        e1, e2, e3 = self.backbone(inputs)[:3]
        e1 = self.fuse1(e1.reshape(batch, slices * e1.shape[1], e1.shape[2], e1.shape[3]))
        e2 = self.fuse2(e2.reshape(batch, slices * e2.shape[1], e2.shape[2], e2.shape[3]))
        e3 = self.fuse3(e3.reshape(batch, slices * e3.shape[1], e3.shape[2], e3.shape[3]))

        c1, c2, c3 = self.c1(e1), self.c2(e2), self.c3(e3)
        d1 = self.d1(c3, c2)
        d2 = self.d2(d1, c1)
        d3 = self.d3(d2)
        features = torch.cat([d3, self.u1(c1), self.u2(c2), self.u3(c3)], dim=1)
        return self.output(self.fuse_output(features))
