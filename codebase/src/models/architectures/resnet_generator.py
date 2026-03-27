import torch
import torch.nn as nn
from utils import count_parameters

import scripting
from core import get_logger


class ConvBlock(nn.Module):
    """
    A convolutional block used in the ResNet generator, consisting of a
    convolutional layer, instance normalization, and a ReLU activation.

    Attributes:
        conv (nn.Sequential): The sequential container for the block's layers.
    """
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        """
        Initialize the ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            down (bool, optional): Whether to use a Conv2d (downsampling) or
                                   ConvTranspose2d (upsampling). Defaults to True.
            use_act (bool, optional): Whether to include the ReLU activation. Defaults to True.
            **kwargs: Additional keyword arguments for the convolutional layer.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        """
        Forward pass through the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    A residual block that adds the input to the output of two convolutional blocks.
    This is a key component of the ResNet architecture.

    Attributes:
        block (nn.Sequential): The sequential container for the two ConvBlocks.
    """
    def __init__(self, channels):
        """
        Initialize the ResidualBlock.

        Args:
            channels (int): The number of input and output channels for the block.
        """
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Forward pass through the ResidualBlock with a skip connection.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after adding the input (skip connection).
        """
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    A ResNet-based generator model, as used in the CycleGAN paper.
    It consists of an initial convolutional layer, downsampling blocks,
    a series of residual blocks, upsampling blocks, and a final output layer.

    Attributes:
        initial (nn.Sequential): The initial convolutional block.
        down_blocks (nn.ModuleList): List of downsampling blocks.
        res_blocks (nn.Sequential): A sequence of residual blocks.
        up_blocks (nn.ModuleList): List of upsampling blocks.
        last (nn.Sequential): The final output layer with a Sigmoid activation.
    """
    def __init__(self, in_channels=1, features=64, num_residuals=9, normalization_type='range_zero_to_one'):
        """
        Initialize the GeneratorResNet.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1.
            features (int, optional): Base number of features. Defaults to 64.
            num_residuals (int, optional): Number of residual blocks. Defaults to 9.
            normalization_type (str, optional): Type of normalization, determines activation. Defaults to 'range_zero_to_one'.
        """
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    features, features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    features * 2,
                    features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(features * 4) for _ in range(num_residuals)])
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    features * 4,
                    features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    features * 2,
                    features,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        if normalization_type == 'range_minus_one_to_one':
            final_activation = nn.Tanh()
        else:
            final_activation = nn.Sigmoid()

        self.last = nn.Sequential(
            nn.Conv2d(
                features,
                in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            final_activation
        )

        get_logger().debug(f"Total <generator> learnable parameters: {count_parameters(self)}")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return self.last(x)


def test():
    """
    Test the ResnetGenerator model with a random input tensor.

    Behavior:
        - Creates a random input tensor of shape (1, 1, 256, 256).
        - Initializes the GeneratorResNet model.
        - Passes the input through the model and prints the input and output shapes.
    """
    logger = get_logger()
    logger.info("Testing ResNet Generator...")
    in_channels = 1
    img_size = 256
    x = torch.randn((1, in_channels, img_size, img_size))
    model = ResnetGenerator(in_channels=in_channels, features=64)
    preds = model(x)
    logger.info("ResNet Generator test completed.")
    logger.info(f"Input shape: {x.shape}, Output shape: {preds.shape}")

if __name__ == "__main__":
    """
    Entry point for testing the ResnetGenerator model.
    
    Behavior:
        - Calls the test function to run a test with a random input tensor.
    """
    scripting.logged_main(
        "Testing ResNet Generator",
        test,
    )