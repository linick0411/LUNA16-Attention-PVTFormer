import torch
import torch.nn as nn
from pvtv2 import pvt_v2_b3
from model_utils import load_pvtv2_b3_weights


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=True, act=True):
        super().__init__()

        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x


class DecoderBlock(nn.Module):
    """Simple Decoder: upsample + concat(skip) + ResidualBlock (no attention gate)"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c + out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], dim=1)
        x = self.r1(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c, out_c)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.r1(x)
        return x


class CoordinateAttentionFuse(nn.Module):
    """
    Coordinate Attention-based feature fusion for 2.5D slices.
    Fuses features from 3 slices (B, S*C_in, H, W) -> (B, C_in, H, W)
    using spatial attention along H and W axes independently.
    Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021.
    """
    def __init__(self, C_in, num_slices=3, reduction=32):
        super().__init__()
        C_cat = C_in * num_slices
        C_mid = max(C_cat // reduction, 8)

        # Shared MLP to compress the concatenated H+W pooled features
        self.conv_reduce = nn.Conv2d(C_cat, C_mid, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(C_mid)
        self.act = nn.ReLU(inplace=True)

        # Separate 1x1 convs to generate H-attention and W-attention maps
        self.conv_h = nn.Conv2d(C_mid, C_in, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(C_mid, C_in, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Final conv to project S*C_in channels down to C_in
        self.fuse = nn.Conv2d(C_cat, C_in, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, S*C, H, W)
        B, C_cat, H, W = x.shape

        # Pool along W -> (B, C_cat, H, 1)
        x_h = x.mean(dim=3, keepdim=True)
        # Pool along H -> (B, C_cat, 1, W) then transpose to (B, C_cat, W, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        # Concat along spatial dim: (B, C_cat, H+W, 1)
        x_cat = torch.cat([x_h, x_w], dim=2)

        # Shared reduction
        x_cat = self.act(self.bn(self.conv_reduce(x_cat)))  # (B, C_mid, H+W, 1)

        # Split back
        x_h_attn = x_cat[:, :, :H, :]       # (B, C_mid, H, 1)
        x_w_attn = x_cat[:, :, H:, :]       # (B, C_mid, W, 1)

        # Generate attention maps
        a_h = self.sigmoid(self.conv_h(x_h_attn))           # (B, C_in, H, 1)
        a_w = self.sigmoid(self.conv_w(x_w_attn)).permute(0, 1, 3, 2)  # (B, C_in, 1, W)

        # Project to C_in channels, then apply H & W attention
        x_fused = self.fuse(x)       # (B, C_in, H, W)
        out = x_fused * a_h * a_w    # broadcast: (B, C_in, H, W)
        return out


class PVTFormer(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.backbone = pvt_v2_b3()
        load_pvtv2_b3_weights(self.backbone)

        """ Feature Fusion (Coordinate Attention) """
        # Fuse the 3 slice features at e1, e2, e3 level with Coordinate Attention
        self.fuse1 = CoordinateAttentionFuse(64,  num_slices=3)
        self.fuse2 = CoordinateAttentionFuse(128, num_slices=3)
        self.fuse3 = CoordinateAttentionFuse(320, num_slices=3)

        """ Channel Reduction (都壓到 64 channel，後面好接) """
        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)

        """ Standard Decoder (no Attention Gate) """
        self.d1 = DecoderBlock(64, 64)   # c3 + c2
        self.d2 = DecoderBlock(64, 64)   # d1 + c1
        self.d3 = UpBlock(64, 64, 4)     # H/4 -> H

        # 三條直接從 encoder 各 stage 拉回原圖大小
        self.u1 = UpBlock(64, 64, 4)     # c1: H/4  -> H
        self.u2 = UpBlock(64, 64, 8)     # c2: H/8  -> H
        self.u3 = UpBlock(64, 64, 16)    # c3: H/16 -> H

        # 四個 64-channel feature concat 後 (256)，再壓回 64
        self.r1 = ResidualBlock(64 * 4, 64)
        self.y = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        # inputs shape for 2.5D: (B, S=3, C=1, H, W)
        if inputs.ndim == 5:
            B, S, C, H, W = inputs.shape
            if C == 1:
                inputs_2d = inputs.repeat(1, 1, 3, 1, 1) # (B, S, 3, H, W)
            else:
                inputs_2d = inputs
            inputs_2d = inputs_2d.view(B * S, inputs_2d.shape[2], H, W)
        else:
            inputs_2d = inputs
            B = inputs.shape[0]
            S = 1

        """ Encoder (Siamese processing) """
        pvt1 = self.backbone(inputs_2d)
        e1 = pvt1[0]     # (B*S, 64, H/4, W/4)
        e2 = pvt1[1]     # (B*S, 128, H/8, W/8)
        e3 = pvt1[2]     # (B*S, 320, H/16, W/16)

        if S > 1:
            # Reshape e1, e2, e3 to concatenate the slice features in the channel dimension
            # from (B*S, C, H, W) to (B, S*C, H, W)
            e1 = e1.view(B, S * e1.shape[1], e1.shape[2], e1.shape[3])
            e2 = e2.view(B, S * e2.shape[1], e2.shape[2], e2.shape[3])
            e3 = e3.view(B, S * e3.shape[1], e3.shape[2], e3.shape[3])
            
            # Simple 1x1 Conv feature fusion (No Attention)
            e1 = self.fuse1(e1)
            e2 = self.fuse2(e2)
            e3 = self.fuse3(e3)

        # 壓到同一個 channel 數
        c1 = self.c1(e1)  # [B, 64, H/4,  W/4]
        c2 = self.c2(e2)  # [B, 64, H/8,  W/8]
        c3 = self.c3(e3)  # [B, 64, H/16, W/16]

        """ Standard Decoder (no AG) """
        d1 = self.d1(c3, c2)
        d2 = self.d2(d1, c1)
        d3 = self.d3(d2)

        """ Encoder 多尺度分支拉回原圖 """
        u1 = self.u1(c1)      # H/4  -> H
        u2 = self.u2(c2)      # H/8  -> H
        u3 = self.u3(c3)      # H/16 -> H

        """ 多尺度融合 """
        x = torch.cat([d3, u1, u2, u3], dim=1)  # [B, 256, H, W]
        x = self.r1(x)                          # [B, 64,  H, W]
        y = self.y(x)                           # [B, 1,   H, W]
        return y


if __name__ == "__main__":
    print("Testing 2.5D Input on PVTFormer with CoordinateAttention fusion + standard decoder...")
    x25d = torch.randn((2, 3, 1, 256, 256)) 
    model = PVTFormer()
    y = model(x25d)
    print(f"Output Config Shape: {y.shape}")
