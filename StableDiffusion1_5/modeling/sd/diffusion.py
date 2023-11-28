import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

from modeling.attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    """
    Time Embedding Module for adding time-dependent features to a neural network.

    Attributes:
        linear_1 (nn.Linear): First linear layer.
        linear_2 (nn.Linear): Second linear layer.
    """

    def __init__(self, n_embd: int, linear_dim: int = 4):
        """
        Initialize the TimeEmbedding module.

        Args:
            n_embd (int): Dimension of the input embedding.
            linear_dim (int, optional): Multiplier for the dimension of the linear layers. Defaults to 4.
        """
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, linear_dim * n_embd)
        self.linear_2 = nn.Linear(linear_dim * n_embd, linear_dim * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TimeEmbedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, n_embd).

        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, linear_dim * n_embd).
        """
        # (Batch_Size, n_embd) -> (Batch_Size, linear_dim * n_embd)
        x = self.linear_1(x)

        # (Batch_Size, linear_dim * n_embd) -> (Batch_Size, linear_dim * n_embd)
        x = F.silu(x)

        # (Batch_Size, linear_dim * n_embd) -> (Batch_Size, linear_dim * n_embd)
        x = self.linear_2(x)

        return x


class UNETResidualBlock(nn.Module):
    """
    UNET Residual Block for use in a UNET architecture.

    Attributes:
        groupnorm_feature (nn.GroupNorm): Group normalization layer for features.
        conv_feature (nn.Conv2d): Convolution layer for features.
        linear_time (nn.Linear): Linear layer for time embeddings.
        groupnorm_merged (nn.GroupNorm): Group normalization layer for merged tensor.
        conv_merged (nn.Conv2d): Convolution layer for merged tensor.
        residual_layer (nn.Module): Residual layer, either Identity or Conv2d.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_time: int = 1280,
            num_groups: int = 32
    ):
        """
        Initialize the UNETResidualBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_time (int, optional): Dimension of the time embedding. Defaults to 1280.
            num_groups (int, optional): Number of groups for GroupNorm. Defaults to 32.
        """
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(num_groups, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(num_groups, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UNETResidualBlock module.

        Args:
            feature (torch.Tensor): Input feature tensor of shape (Batch_Size, In_Channels, Height, Width).
            time (torch.Tensor): Time tensor of shape (Batch_Size, n_time).

        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Out_Channels, Height, Width).
        """
        # Store the original feature for the residual connection
        residue = feature

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)

        # (Batch_Size, n_time) -> (Batch_Size, n_time)
        time = F.silu(time)

        # (Batch_Size, n_time) -> (Batch_Size, Out_Channels)
        time = self.linear_time(time)

        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)

        # Return the sum of the residual and the merged tensor
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)

class UNETAttentionBlock(nn.Module):
    """An attention block for U-Net architecture, consisting of Self-Attention, Cross-Attention, and a FFN with GeGLU.

    Attributes:
        groupnorm (nn.GroupNorm): Normalization layer before input convolution.
        conv_input (nn.Conv2d): Convolution layer for input tensor.
        layernorm_1 (nn.LayerNorm): Layer normalization before first self-attention layer.
        attention_1 (SelfAttention): First self-attention layer.
        layernorm_2 (nn.LayerNorm): Layer normalization before cross-attention layer.
        attention_2 (CrossAttention): Cross-attention layer.
        layernorm_3 (nn.LayerNorm): Layer normalization before FFN with GeGLU.
        linear_geglu_1 (nn.Linear): First linear layer in FFN with GeGLU.
        linear_geglu_2 (nn.Linear): Second linear layer in FFN with GeGLU.
        conv_output (nn.Conv2d): Convolution layer for output tensor.
    """

    def __init__(self, n_head: int, n_embd: int, d_context: int = 768, groupnorm_eps: float = 1e-6,
                 groupnorm_num_groups: int = 32, in_proj_bias: bool = False):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(groupnorm_num_groups, channels, eps=groupnorm_eps)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=in_proj_bias)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=in_proj_bias)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass for the attention block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, features, height, width).
            context (Tensor): Context tensor of shape (batch_size, seq_len, dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, features, height, width).
        """
        # Long residual connection
        residue_long = x

        # Normalize input
        x = self.groupnorm(x)

        # Convolution to adjust the input channels
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # Reshape for attention operations
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        # Short residual connection before Self-Attention
        residue_short = x

        # Layer normalization and Self-Attention
        x = self.layernorm_1(x)
        x = self.attention_1(x)

        # Add short residual connection
        x += residue_short

        # Short residual connection before Cross-Attention
        residue_short = x

        # Layer normalization and Cross-Attention
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)

        # Add short residual connection
        x += residue_short

        # Short residual connection before FFN
        residue_short = x

        # Layer normalization and FFN with GeGLU
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)

        # Add short residual connection
        x += residue_short

        # Reshape back to original dimensions
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        # Final skip connection and output
        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    """Upsampling block using nearest-neighbor interpolation followed by a convolution layer.

    Attributes:
        conv (nn.Conv2d): Convolution layer for upsampled tensor.
    """

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the upsampling block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, features, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, features, height * 2, width * 2).
        """
        # Upsample using nearest-neighbor interpolation
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # Convolution to adjust the channels
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """Custom sequential module that supports the custom U-Net attention block and other modules.

    This class inherits from nn.Sequential and overrides the forward method to support custom blocks.
    """

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for the SwitchSequential module.

        Args:
            x (Tensor): Input tensor.
            context (Tensor, optional): Context tensor for attention blocks.
            time (Tensor, optional): Time tensor for residual blocks.

        Returns:
            Tensor: Output tensor.
        """
        for layer in self:
            if isinstance(layer, UNETAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNETResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    """
    UNET Class for image segmentation.

    Attributes:
        encoders (nn.ModuleList): A list of encoder layers.
        bottleneck (nn.Module): The bottleneck layer.
        decoders (nn.ModuleList): A list of decoder layers.
    """
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNETResidualBlock(320, 640), UNETAttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNETResidualBlock(640, 640), UNETAttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNETResidualBlock(640, 1280), UNETAttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNETResidualBlock(1280, 1280), UNETAttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNETResidualBlock(1280, 1280)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNETResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNETResidualBlock(1280, 1280),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNETAttentionBlock(8, 160),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNETResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNETResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNETResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNETResidualBlock(2560, 1280), Upsample(1280)),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)),

            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNETResidualBlock(1920, 1280), UNETAttentionBlock(8, 160), Upsample(1280)),

            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNETResidualBlock(1920, 640), UNETAttentionBlock(8, 80)),

            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNETResidualBlock(1280, 640), UNETAttentionBlock(8, 80)),

            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNETResidualBlock(960, 640), UNETAttentionBlock(8, 80), Upsample(640)),

            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNETResidualBlock(960, 320), UNETAttentionBlock(8, 40)),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNET_OutputLayer(nn.Module):
    """
    Diffusion output layer.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)

        # (Batch_Size, 4, Height / 8, Width / 8)
        return x


class Diffusion(nn.Module):
    """
    Diffusion model
    """
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)

        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)

        # (Batch, 4, Height / 8, Width / 8)
        return output