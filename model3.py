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


class AttentionGate(nn.Module):
    """
    Attention U-Net style gate，用來在 skip connection 上篩選 encoder feature
    g: decoder feature (gating)
    x: encoder skip feature
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # 將 decoder feature 壓到中間維度
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 將 encoder skip feature 壓到中間維度
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 輸出 1-channel 的空間權重 (H, W)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: decoder feature (B, F_g, H', W')
        x: encoder feature (B, F_l, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 尺寸不合就對齊一下（理論上你的架構會對齊，但保險）
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = nn.functional.interpolate(
                g1, size=x1.shape[-2:], mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)   # (B, F_int, H, W)
        psi = self.psi(psi)        # (B, 1, H, W)，值在 0~1 之間
        out = x * psi              # 對 skip feature 做空間加權
        return out


class DecoderBlock(nn.Module):
    """
    原本的 DecoderBlock: upsample + concat(skip) + ResidualBlock
    現在版本多了一個 AttentionGate，先用 decoder feature 去 gate skip feature，再 concat。
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # in_c: 來自上一層 decoder 的 channel
        # out_c: skip connection (c1/c2) 的 channel
        F_int = max(out_c // 2, 1)
        self.att = AttentionGate(F_g=in_c, F_l=out_c, F_int=F_int)

        # concat 之後的 channel = in_c + out_c
        self.r1 = ResidualBlock(in_c + out_c, out_c)

    def forward(self, x, s):
        """
        x: decoder 較深一層的 feature
        s: encoder shortcut feature
        """
        x = self.up(x)      # 上採樣到跟 s 差不多大小
        s = self.att(x, s)  # 用 x 當 gate，篩選 s
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


class PVTFormer(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.backbone = pvt_v2_b3()
        load_pvtv2_b3_weights(self.backbone)

        """ Feature Fusion (Concatenation + 1x1 Conv) """
        # Fuse the representations of 3 slices at e1, e2, e3 level without attention
        self.fuse1 = nn.Conv2d(64 * 3, 64, kernel_size=1)
        self.fuse2 = nn.Conv2d(128 * 3, 128, kernel_size=1)
        self.fuse3 = nn.Conv2d(320 * 3, 320, kernel_size=1)

        """ Channel Reduction (都壓到 64 channel，後面好接) """
        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)

        """ Decoder + Multi-scale """
        # 這兩層 decoder 現在是「帶 Attention Gate 的 U-Net decoder」
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

        """ Decoder 主幹（帶 Attention Gate 的 skip connection） """
        d1 = self.d1(c3, c2)  # H/16 -> H/8，gate c2
        d2 = self.d2(d1, c1)  # H/8  -> H/4，gate c1
        d3 = self.d3(d2)      # H/4  -> H

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
    print("Testing 2.5D Sequence Input on PVTFormer with AttentionGate...")
    # B=2, Slices=3, C=1, H=256, W=256
    x25d = torch.randn((2, 3, 1, 256, 256)) 
    model = PVTFormer()
    y = model(x25d)
    print(f"Output Config Shape: {y.shape}")
