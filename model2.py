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


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c, out_c)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.r1(x)
        return x


class VoxelAttention(nn.Module):
    """
    Feature-level Voxel Attention for 2.5D medical imaging.
    Takes independent features from multiple slices, calculates spatial attention weights
    across slices, and fuses them into a single comprehensive feature map.
    """
    def __init__(self, in_channels, num_slices=3):
        super().__init__()
        # 1x1 convolution to calculate cross-slice channel compression and attention scores
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * num_slices, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_slices, kernel_size=1),
            nn.Softmax(dim=1) # Softmax over the Slices dimension
        )
        # Final feature fusion
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        # x shape: (B, S, C, H, W) where S = num_slices = 3
        B, S, C, H, W = x.shape
        # reshaping to pass through 2D Conv
        x_reshaped = x.view(B, S*C, H, W)
        
        # Calculate attention scores: (B, S, H, W)
        attn = self.attention(x_reshaped) 
        
        # Apply attention scores back to the features
        attn_unsqueezed = attn.unsqueeze(2) # (B, S, 1, H, W)
        x_attended = x * attn_unsqueezed
        
        # Sum across slices -> (B, C, H, W)
        x_fused = x_attended.sum(dim=1)
        
        return self.fusion(x_fused)


class PVTFormer_Voxel(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.backbone = pvt_v2_b3()
        load_pvtv2_b3_weights(self.backbone)

        """ Voxel Attention Heads """
        # Fuse the representations of 3 slices at e1, e2, e3 level
        self.va1 = VoxelAttention(64, num_slices=3)
        self.va2 = VoxelAttention(128, num_slices=3)
        self.va3 = VoxelAttention(320, num_slices=3)

        """ Channel Reduction """
        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)

        """ Standard Decoder (No CBAM) """
        self.up_d1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r_d1 = ResidualBlock(64 + 64, 64)
        
        self.up_d2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r_d2 = ResidualBlock(64 + 64, 64)
        
        self.d3 = UpBlock(64, 64, 4)

        self.u1 = UpBlock(64, 64, 4)
        self.u2 = UpBlock(64, 64, 8)
        self.u3 = UpBlock(64, 64, 16)

        self.r1 = ResidualBlock(64 * 4, 64)
        
        """ Final Output """
        self.y = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        # inputs shape for 2.5D: (B, S=3, C=1, H, W)
        
        if inputs.ndim == 5:
            B, S, C, H, W = inputs.shape
            # PVTv2 expects 3 channel RGB images, repeat 1-channel to 3
            if C == 1:
                inputs_2d = inputs.repeat(1, 1, 3, 1, 1) # (B, S, 3, H, W)
            else:
                inputs_2d = inputs
            # Flatten B and S to batch-process all slices through Siamese Backbone
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
            # Reshape back to (B, S, C_out, H_out, W_out)
            e1 = e1.view(B, S, e1.shape[1], e1.shape[2], e1.shape[3])
            e2 = e2.view(B, S, e2.shape[1], e2.shape[2], e2.shape[3])
            e3 = e3.view(B, S, e3.shape[1], e3.shape[2], e3.shape[3])
            
            # Fuse with Voxel Attention
            e1 = self.va1(e1) # (B, 64, H/4, W/4)
            e2 = self.va2(e2) # (B, 128, H/8, W/8)
            e3 = self.va3(e3) # (B, 320, H/16, W/16)

        """ Decoder routing """
        c1 = self.c1(e1)
        c2 = self.c2(e2)
        c3 = self.c3(e3)

        x_up1 = self.up_d1(c3)
        d1 = self.r_d1(torch.cat([x_up1, c2], dim=1))
        
        x_up2 = self.up_d2(d1)
        d2 = self.r_d2(torch.cat([x_up2, c1], dim=1))
        
        d3 = self.d3(d2)

        u1 = self.u1(c1)
        u2 = self.u2(c2)
        u3 = self.u3(c3)

        x = torch.cat([d3, u1, u2, u3], dim=1)
        x = self.r1(x)
        y = self.y(x)
        
        return y


if __name__ == "__main__":
    print("Testing 2.5D Sequence Input on Pure PVTFormer_Voxel...")
    # B=2, Slices=3, C=1, H=256, W=256
    x25d = torch.randn((2, 3, 1, 256, 256)) 
    model_voxel = PVTFormer_Voxel()
    y_voxel = model_voxel(x25d)
    print(f"Voxel Attention Config Output: {y_voxel.shape}")
    
