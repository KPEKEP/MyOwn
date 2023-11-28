import torch
from torch import nn, Tensor
from torch.nn import functional as F

from modeling.vae.decoder import VAEAttentionBlock, VAEResidualBlock


class VAEEncoder(nn.Sequential):
    """VAE Encoder class

    This class is responsible for encoding an input image into a latent space.
    It uses a series of Convolutional, Residual, and Attention blocks.

    Attributes:
        layers (nn.ModuleList): A list of layers that make up the encoder.
    """

    def __init__(self,
                 input_channels: int = 3,
                 initial_conv_channels: int = 128,
                 log_variance_clamp_min: float = -30,
                 log_variance_clamp_max: float = 20,
                 scale_constant: float = 0.18215):
        """Initialize the VAE_Encoder class

        Args:
            input_channels (int): Number of input channels in the image.
            initial_conv_channels (int): Number of output channels for the initial Conv layer.
            log_variance_clamp_min (float): Minimum value to clamp the log variance to.
            log_variance_clamp_max (float): Maximum value to clamp the log variance to.
            scale_constant (float): Constant by which to scale the output.
        """
        self.log_variance_clamp_min = log_variance_clamp_min
        self.log_variance_clamp_max = log_variance_clamp_max
        self.scale_constant = scale_constant

        # Initialize layers
        super().__init__(
            nn.Conv2d(input_channels, initial_conv_channels, kernel_size=3, padding=1),  # Initial Conv layer
            VAEResidualBlock(initial_conv_channels, initial_conv_channels),  # Residual Blocks
            VAEResidualBlock(initial_conv_channels, initial_conv_channels),
            nn.Conv2d(initial_conv_channels, initial_conv_channels, kernel_size=3, stride=2, padding=0),  # Downsample
            VAEResidualBlock(initial_conv_channels, 2 * initial_conv_channels),  # Increase channels
            VAEResidualBlock(2 * initial_conv_channels, 2 * initial_conv_channels),
            nn.Conv2d(2 * initial_conv_channels, 2 * initial_conv_channels, kernel_size=3, stride=2, padding=0),
            VAEResidualBlock(2 * initial_conv_channels, 4 * initial_conv_channels),
            VAEResidualBlock(4 * initial_conv_channels, 4 * initial_conv_channels),
            nn.Conv2d(4 * initial_conv_channels, 4 * initial_conv_channels, kernel_size=3, stride=2, padding=0),
            VAEResidualBlock(4 * initial_conv_channels, 4 * initial_conv_channels),
            VAEResidualBlock(4 * initial_conv_channels, 4 * initial_conv_channels),
            VAEResidualBlock(4 * initial_conv_channels, 4 * initial_conv_channels),
            VAEAttentionBlock(4 * initial_conv_channels),
            VAEResidualBlock(4 * initial_conv_channels, 4 * initial_conv_channels),
            nn.GroupNorm(32, 4 * initial_conv_channels),
            nn.SiLU(),
            nn.Conv2d(4 * initial_conv_channels, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: Tensor, noise: Tensor) -> Tensor:
        """Forward pass of the encoder

        Args:
            x (Tensor): Input image tensor (Batch_Size, Channel, Height, Width)
            noise (Tensor): Noise tensor (Batch_Size, 4, Height / 8, Width / 8)

        Returns:
            Tensor: The encoded tensor
        """

        for module in self.children():

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric
                x = F.pad(x, (0, 1, 0, 1))  # Padding

            x = module(x)

        # Splitting mean and log_variance
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamping log_variance
        log_variance = torch.clamp(log_variance, self.log_variance_clamp_min, self.log_variance_clamp_max)

        # Calculating variance and standard deviation
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # Sampling from N(mean, stdev)
        x = mean + stdev * noise

        # Scaling by constant
        x *= self.scale_constant

        return x