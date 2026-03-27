import scripting
import torch
import torch.nn as nn
from utils import count_parameters
from core import get_logger


class Block(nn.Module):
    """
    A building block for the generator model, which can perform either downsampling
    or upsampling operations with optional dropout.

    Attributes:
        conv (nn.Sequential): The convolutional layer(s) for the block.
        use_dropout (bool): Whether to apply dropout.
        dropout (nn.Dropout): Dropout layer (if enabled).
        down (bool): Whether the block performs downsampling.
    """

    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        """
        Initialize the Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            down (bool, optional): Whether the block performs downsampling. Defaults to True.
            act (str, optional): Activation function ("relu" or "leaky"). Defaults to "relu".
            use_dropout (bool, optional): Whether to apply dropout. Defaults to False.
        """
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            (
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    4,
                    2,
                    1,
                    bias=False,
                    padding_mode="reflect",
                )
                if down
                else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the block operations.
        """
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


import math

class UnetGenerator(nn.Module):
    """
    U-Net Generator Architecture, commonly used for Pix2Pix.
    
    Attributes:
        down (nn.ModuleList): List of downsampling layers.
        up (nn.ModuleList): List of upsampling layers.
    """
    def __init__(self, in_channels=1, features=64, patch_size=256, normalization_type='range_zero_to_one'):
        """
        Initialize the Generator.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1.
            features (int, optional): Base number of features for the model. Defaults to 64.
            patch_size (int, optional): The size of the input patch. Defaults to 256.
            normalization_type (str, optional): Type of normalization, determines activation. Defaults to 'range_zero_to_one'.
        """
        super().__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        # This list will store the output channel counts of each downsampling layer.
        # This will be used for dynamically configuring the UP path's skip connections.
        down_out_channels = []
        
        num_down_layers = int(math.log2(patch_size))
        
        down_layers_config = []
        
        # Initial layer
        current_channels = in_channels
        next_channels = features
        down_layers_config.append((current_channels, next_channels, False, "leaky"))
        current_channels = next_channels

        # Downsampling blocks
        for _ in range(num_down_layers - 2): # Subtract initial and bottleneck layers
            next_channels = min(current_channels * 2, features * 8)
            down_layers_config.append((current_channels, next_channels, True, "leaky"))
            current_channels = next_channels

        # Bottleneck layer
        down_layers_config.append((current_channels, min(current_channels * 2, features * 8), False, "relu"))

        for in_c, out_c, use_block, act in down_layers_config:
            if use_block:
                layer = Block(in_c, out_c, down=True, act=act, use_dropout=False)
            else:
                layer = nn.Sequential(
                            nn.Conv2d(
                                in_c, out_c, kernel_size=4, stride=2, padding=1,
                                padding_mode="reflect" if act == "leaky" else "zeros"
                            ),
                            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
                        )
            self.down.append(layer)
            # Store the output channel count of each downsampling layer.
            down_out_channels.append(out_c)

        # --- Upsampling Path Construction ---
        self.up = nn.ModuleList()

        # Channel count of the feature maps from the down-path, to be used for skip connections.
        # We reverse the list to go from the bottleneck up to the input layer.
        skip_connection_channels = down_out_channels[::-1]
        
        # The up-path starts with the channels from the bottleneck layer.
        up_path_ch_count = skip_connection_channels[0]

        # We iterate from the layer after the bottleneck up to the layer just before the output.
        # The range is `len(skip_connection_channels) - 1` which is `num_down_layers - 1`.
        for i in range(len(skip_connection_channels) - 1):
            use_dropout = i < 3  # Apply dropout to the 3 layers closest to the bottleneck.

            # Determine the input channels for the current up-sampling block.
            # For the first block (i=0), the input is just the bottleneck's output.
            # For subsequent blocks, the input is the concatenation of the previous up-block's
            # output and the corresponding skip connection from the down-path.
            # The logic correctly deduces that the skip connection's channel count
            # is the same as the previous up-block's output channel count.
            input_ch_count = up_path_ch_count if i == 0 else up_path_ch_count * 2

            # The output channels of this block are set to match the next skip connection's channels.
            output_ch_count = skip_connection_channels[i + 1]
            
            self.up.append(Block(
                input_ch_count,
                output_ch_count,
                down=False,
                act="relu",
                use_dropout=use_dropout
            ))
            
            # The output of the current block becomes the input for the next one in the up-path stream.
            up_path_ch_count = output_ch_count

        # --- Final Upsampling Layer ---
        # This layer brings the feature map back to the original number of input channels.
        # Its input is a concatenation of the last up-block's output and the very first down-block's output.
        last_up_block_channels = up_path_ch_count
        first_down_block_channels = skip_connection_channels[-1] # This is down_out_channels[0]
        
        # Determine the final activation function based on the normalization type
        if normalization_type == 'range_minus_one_to_one':
            final_activation = nn.Tanh()
        else: # Default to sigmoid for [0, 1] or other cases
            final_activation = nn.Sigmoid()

        self.up.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    last_up_block_channels + first_down_block_channels,
                    in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                final_activation
            )
        )
        
        get_logger().debug(f"Total <generator> learnable parameters: {count_parameters(self)}")

    def forward(self, x):
        """
        Forward pass through the generator.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor after applying the generator operations.
        """
        data = x
        down_res = []

        for idx, layer in enumerate(self.down):
            data = layer(data)
            down_res.append(data)

        data = self.up[0](down_res[-1])  # Start with the last downsampled layer (bottleneck)
        
        # Iterate through the upsampling layers, concatenating with skip connections
        for i, layer in enumerate(self.up[1:-1]):
            skip = down_res[-(i + 2)]
            data = layer(
                torch.cat([data, skip], 1)
            )  # concatenate with skip connection

        # Final upsampling layer
        data = self.up[-1](torch.cat([data, down_res[0]], 1))
        
        return data
    
    
def test():
    """
    Test the Generator model with a random input tensor.

    Behavior:
        - Creates a random input tensor with shape (1, 1, 256, 256).
        - Initializes the Generator model.
        - Passes the input tensor through the model.
        - Prints the shape of the output tensor.
    """
    logger = get_logger()
    logger.info("Testing U-Net Generator...")
    x = torch.randn((1, 1, 256, 256))
    model = UnetGenerator(in_channels=1, features=16)
    preds = model(x)
    logger.info(f"Output shape: {preds.shape}")


if __name__ == "__main__":
    scripting.logged_main(
        "U-Net Generator",
        test,
    )
