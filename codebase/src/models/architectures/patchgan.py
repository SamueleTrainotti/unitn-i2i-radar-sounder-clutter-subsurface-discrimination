import torch
import torch.nn as nn

import scripting
from core import get_logger
from utils import count_parameters


class CNNBlock(nn.Module):
    """
    A convolutional block used in the discriminator model.

    Attributes:
        conv (nn.Sequential): The convolutional layer(s) for the block.
    """
    def __init__(self, in_channels, out_channels, stride):
        """
        Initialize the CNNBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolutional layer.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the CNNBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the block operations.
        """
        return self.conv(x)


class PatchGAN(nn.Module):
    """
    PatchGAN Discriminator, used by both Pix2Pix and CycleGAN.
    It tries to classify if each NxN patch in an image is real or fake.
    
    Attributes:
        initial (nn.Sequential): Initial convolutional layer for the discriminator.
        model (nn.Sequential): Sequential layers of the discriminator.
    """
    def __init__(self, in_channels=1, receptive_field="70x70"):
        """
        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1.
            receptive_field (str, optional): Receptive field of the discriminator.
                                             Supported values: "70x70", "34x34", "16x16".
                                             Defaults to "70x70".
        """
        super().__init__()

        if receptive_field == "70x70":
            features = (64, 128, 256, 512)
        elif receptive_field == "34x34":
            features = (64, 128, 256)
        elif receptive_field == "16x16":
            features = (64, 128)
        else:
            raise ValueError(f"Unsupported receptive field: {receptive_field}")

        # The input to the discriminator will be a single image for CycleGAN,
        # or a concatenated pair of images for Pix2Pix. This is handled
        # in the respective model's training logic.
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels_ = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels_, feature, stride=1 if feature == features[-1] else 2))
            in_channels_ = feature

        layers.append(nn.Conv2d(in_channels_, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)
        
        get_logger().debug(f"Total <discriminator> learnable parameters: {count_parameters(self)}")

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input tensor, typically a concatenation of input and target images.

        Returns:
            torch.Tensor: Output tensor representing the discriminator's prediction.
        """
        x = self.initial(x)
        return self.model(x)
    
    
def test():
    """
    Test the Discriminator model with random input tensors.

    Behavior:
        - Creates random input tensors `x` and `y` with shape (1, 1, 256, 256).
        - Initializes the Discriminator model.
        - Passes the input tensors through the model.
        - Prints the model architecture and the shape of the output tensor.
    """
    logger = get_logger()
    logger.info("Testing PatchGAN Discriminator...")
    x = torch.randn((1, 1, 256, 256))
    y = torch.randn((1, 1, 256, 256))
    model = PatchGAN(in_channels=2)
    preds = model(torch.cat([x, y], dim=1))
    logger.info(f"Input shape: {x.shape}, Output shape: {preds.shape}")
    logger.info("PatchGAN Discriminator test completed.")
        
    
if __name__ == "__main__":
    """
    Entry point for testing the PatchGAN model.
    
    Behavior:
        - Calls the test function to run a test with random input tensors.
    """
    scripting.logged_main(
        "Testing PatchGAN Discriminator",
        test,
    )