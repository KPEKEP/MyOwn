from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from modeling.attention import SelfAttention


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 32) -> None:
        """
        Attention block with Group Normalization for VAE.

        Parameters:
        - channels (int): Number of input and output channels.
        - num_groups (int): Number of groups for Group Normalization. Default is 32.
        """
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention block.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (Batch_Size, Features, Height, Width)

        Returns:
        - torch.Tensor: Output tensor of shape (Batch_Size, Features, Height, Width)
        """
        # Initial shape: (Batch_Size, Features, Height, Width)
        residue = x

        # Shape: (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)

        n, c, h, w = x.shape

        # Shape: (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))

        # Shape: (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # Perform self-attention WITHOUT mask
        # Shape: (Batch_Size, Height * Width, Features)
        x = self.attention(x)

        # Shape: (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # Shape: (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Shape: (Batch_Size, Features, Height, Width)
        x += residue

        # Final shape: (Batch_Size, Features, Height, Width)
        return x


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32) -> None:
        """
        Residual block with Group Normalization for VAE.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - num_groups (int): Number of groups for Group Normalization. Default is 32.
        """
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(num_groups, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(num_groups, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (Batch_Size, In_Channels, Height, Width)

        Returns:
        - torch.Tensor: Output tensor of shape (Batch_Size, Out_Channels, Height, Width)
        """
        # Initial shape: (Batch_Size, In_Channels, Height, Width)
        residue = x

        # Shape: (Batch_Size, In_Channels, Height, Width)
        x = self.groupnorm_1(x)

        # Shape: (Batch_Size, In_Channels, Height, Width)
        x = F.silu(x)

        # Shape: (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_1(x)

        # Shape: (Batch_Size, Out_Channels, Height, Width)
        x = self.groupnorm_2(x)

        # Shape: (Batch_Size, Out_Channels, Height, Width)
        x = F.silu(x)

        # Shape: (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_2(x)

        # Final shape: (Batch_Size, Out_Channels, Height, Width)
        return x + self.residual_layer(residue)


class VAEDecoder(nn.Sequential):
    def __init__(self, scale_factor: Optional[float] = 0.18215) -> None:
        """
        Decoder module for VAE, consisting of multiple residual and attention blocks, followed by upsampling.

        Parameters:
        - scale_factor (Optional[float]): Scaling factor to adjust the input. Default is 0.18215.
        """
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (Batch_Size, 4, Height / 8, Width / 8)

        Returns:
        - torch.Tensor: Output tensor of shape (Batch_Size, 3, Height, Width)
        """
        # Initial shape: (Batch_Size, 4, Height / 8, Width / 8)
        x /= self.scale_factor

        for module in self:
            x = module(x)

        # Final shape: (Batch_Size, 3, Height, Width)
        return x