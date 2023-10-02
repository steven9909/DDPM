import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels, img_size, num_heads=4):
        super().__init__()

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.channels = in_channels
        self.img_size = img_size
        
        self.mha = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(in_channels)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels is None else out_channels
        self.seq = nn.Sequential(
            [
                nn.Upsample(scale_factor=2, mode="nearest"), 
                nn.Conv2d(self.in_channels, self.out_channels, 3, 1)
            ]
        )
    def forward(self, x):
        return self.seq(x)
    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=256):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, out_channels)
        )

        self.embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels)
        )
    
    def forward(self, x, t):
        x = self.downsample(x)
        time_embedding = self.embed(t)[:, :, None, None]
        return x + time_embedding
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, embed_channels, out_channels, residual=False):
        super().__init__()

        self.residual = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, embed_channels),
            nn.GELU(),
            nn.Conv2D(embed_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv(x))
        else:
            return self.conv(x)


        