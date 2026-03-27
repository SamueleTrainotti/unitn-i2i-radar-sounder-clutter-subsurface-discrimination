import logging
import os

import numpy as np
from core import get_logger, get_config
import torch # type: ignore
import torch.nn as nn # Added for get_gradient_norm
import imageio.v3 as iio # type: ignore
import matplotlib.pyplot as plt
import math # Added for get_gradient_norm


def get_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def save_some_examples(gen, val_loader, epoch, folder, device, tiff=False, num_examples=5):
    """
    Save a grid of generated examples, input images, and labels to a specified folder.

    Args:
        gen (torch.nn.Module): The generator model used to create fake images.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        epoch (int): Current epoch number.
        folder (str): Path to the folder where images will be saved.
        tiff (bool): Whether to save images as TIFF files.
        num_examples (int): Number of examples to save in the grid.
    """
    logger = get_logger()
    
    # Get a few examples
    examples = []
    for i, (x, y) in enumerate(val_loader):
        if i >= num_examples:
            break
        examples.append((x, y))

    gen.eval()
    with torch.no_grad():
        # Create a subplot for the examples
        # Use a cleaner layout: Rows = Samples, Cols = Input | Target | Generated
        fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))
        
        # Handle single example case where axes might be 1D
        if num_examples == 1:
            axes = axes.reshape(1, -1)
            
        for i, (x, y) in enumerate(examples):
            x, y = x.to(device), y.to(device)
            y_fake = gen(x.unsqueeze(1))

            # Squeeze tensors to remove batch/channel dimensions for plotting
            # Normalize to [0, 1] for display if needed, but assuming data is in [-1, 1] or [0, 1]
            # If standard tanh output [-1, 1], map to [0, 1]
            def to_numpy(t):
                t = t.squeeze().detach().cpu()
                if t.min() < 0: # Assume [-1, 1]
                    t = (t + 1) / 2
                return torch.clamp(t, 0, 1).numpy()

            x_plot = to_numpy(x)
            y_plot = to_numpy(y)
            y_fake_plot = to_numpy(y_fake)

            # Plot Input
            axes[i, 0].imshow(x_plot, cmap='gray')
            axes[i, 0].axis('off')
            
            # Plot Target
            axes[i, 1].imshow(y_plot, cmap='gray')
            axes[i, 1].axis('off')
            
            # Plot Generated
            axes[i, 2].imshow(y_fake_plot, cmap='gray')
            axes[i, 2].axis('off')

            # Set titles only for the first row
            if i == 0:
                axes[i, 0].set_title("Input (Sim)", fontsize=14)
                axes[i, 1].set_title("Target (Real)", fontsize=14)
                axes[i, 2].set_title("Generated (Fake)", fontsize=14)
        
        plt.tight_layout()
        output_path = f"{folder}/comparison_epoch_{epoch}.png"
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved example grid to {output_path}")

        # Save distribution plots for the first example
        x, y = examples[0]
        x, y = x.to(device), y.to(device)
        y_fake = gen(x.unsqueeze(1))
        
        with logger.indent():
            logger.debug(f"Saving pixel distributions for epoch {epoch}...")
            save_pixel_distribution(x, filepath=f'{folder}/input_distribution_{epoch}.png', title='Input Image Distribution')
            save_pixel_distribution(y, filepath=f'{folder}/label_distribution_{epoch}.png', title='Label Image Distribution')
            save_pixel_distribution(y_fake, filepath=f'{folder}/generated_distribution_{epoch}.png', title='Generated Image Distribution')
        
        with logger.indent():
            logger.debug(f"Saving comparison distributions for epoch {epoch}...")
            save_compare_distributions(y, y_fake, filepath=f'{folder}/compare_distribution_{epoch}.png')
            save_rangeline_comparison(y, y_fake, filepath=f'{folder}/compare_rangeline_{epoch}.png')

    gen.train()

def get_image_stats(image_tensor):
    """
    Get statistics of an image tensor.
    Args:
        image_tensor (torch.Tensor): Image tensor
    Returns:
        dict: Dictionary containing min, max, mean, std, median, shape, and dtype
    """
    stats = {
        "min": image_tensor.min().item(),
        "max": image_tensor.max().item(),
        "mean": image_tensor.mean().item(),
        "std": image_tensor.std().item(),
        "median": image_tensor.median().item(),
        "shape": tuple(image_tensor.shape),
        "dtype": image_tensor.dtype
    }
    return stats

# Save pixel value distribution as a histogram image.
def save_pixel_distribution(image_tensor, filepath="pixel_distribution.png", title="Pixel Value Distribution"):
    plt.set_loglevel (level = 'warning')
    
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.detach().cpu().numpy()
    else:
        image_np = image_tensor
    if image_np.ndim == 2:  # [H, W]
        image_np = image_np[np.newaxis, ...]  # Add channel dimension

    plt.figure(figsize=(10, 4))
    if image_np.ndim == 3:  # [C, H, W]
        for c in range(image_np.shape[0]):
            plt.hist(image_np[c].flatten(), bins=256, alpha=0.6, label=f'Channel {c}')
        plt.legend()
    else:  # Grayscale [H, W]
        plt.hist(image_np.flatten(), bins=256, alpha=0.8, color='gray')

    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
# Save a comparison of pixel value distributions between two images.
def save_compare_distributions(original, generated, filepath="compare_distribution.png", labels=("Original", "Generated")):
    plt.set_loglevel (level = 'warning')
    
    if isinstance(original, torch.Tensor):
        original_np = original.detach().cpu().numpy()
    else:
        original_np = original
    if isinstance(generated, torch.Tensor):
        generated_np = generated.detach().cpu().numpy()
    else:
        generated_np = generated

    # Ensure both images have the same width and height (last two dimensions)
    if original.shape[-2:] != generated.shape[-2:]:
        logging.warning(f"Shape mismatch: original {original.shape}, generated {generated.shape}")
        
    plt.figure(figsize=(12, 5))
    if original_np.ndim == 2:  # [H, W]
        original_np = original_np[np.newaxis, ...]  # Add channel dimension
    if generated_np.ndim == 2:
        generated_np = generated_np[np.newaxis, ...]
    #(f"Processed Original shape: {original_np.shape}, Processed Generated shape: {generated_np.shape}")
    
    for c in range(original_np.shape[0]):
        # print(f"Processing channel {c} for comparison")
        plt.subplot(1, original_np.shape[0], c+1)
        plt.hist(original_np[c].flatten(), bins=256, alpha=0.5, label=labels[0])
        plt.hist(generated_np[c].flatten(), bins=256, alpha=0.5, label=labels[1])
        plt.title(f'Channel {c}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
def save_rangeline_comparison(original, generated, filepath="compare_rangeline.png", title="Middle Column Intensity Profile"):
    """
    Save a plot comparing the middle column of pixel values from two grayscale images.

    Args:
        original (torch.Tensor): Original image tensor [B, 1, H, W] or [1, H, W]
        generated (torch.Tensor): Generated image tensor of the same shape
        filepath (str): Output path to save the plot
        title (str): Plot title
    """
    original_np = original.squeeze().detach().cpu().numpy()
    generated_np = generated.squeeze().detach().cpu().numpy()

    if original_np.ndim == 3:  # If [C, H, W]
        original_np = original_np[0]  # Assume channel 0
        generated_np = generated_np[0]

    h, w = original_np.shape
    mid_col = w // 2

    orig_line = original_np[:, mid_col]
    gen_line = generated_np[:, mid_col]

    plt.figure(figsize=(8, 5))
    plt.plot(orig_line, label="Original", color="blue")
    plt.plot(gen_line, label="Generated", color="red", linestyle='--')
    plt.title(title)
    plt.xlabel("Row Index (Height)")
    plt.ylabel("Pixel Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_evolution(model, output_path):
    """
    Plot how a single generated sample evolves during training.
    Each row represents a snapshot at a specific epoch.
    Columns show the Input, Target, and Generated images.
    """
    input_img = model._evolution_input_sample
    target_img = model._evolution_target_sample
    pred_history = model._evolution_samples  # This is a list of (epoch, tensor)

    if not pred_history:
        get_logger().warning("No evolution samples to plot.")
        return

    num_epochs_recorded = len(pred_history)
    # Create a grid: rows are epochs, columns are Input, Target, Generated
    fig, axes = plt.subplots(num_epochs_recorded, 3, figsize=(9, 3 * num_epochs_recorded))

    # Handle case of a single recorded epoch for correct axes indexing
    if num_epochs_recorded == 1:
        axes = axes.reshape(1, -1)

    # Squeeze tensors to remove batch/channel dimensions for plotting
    input_plot = input_img.squeeze().cpu().numpy()
    target_plot = target_img.squeeze().cpu().numpy()

    for i, (epoch, pred_tensor) in enumerate(pred_history):
        pred_plot = pred_tensor.squeeze().cpu().numpy()

        # Plot Input, Target, and Generated images
        axes[i, 0].imshow(input_plot, cmap='gray')
        axes[i, 1].imshow(target_plot, cmap='gray')
        axes[i, 2].imshow(pred_plot, cmap='gray')

        # Set row label (epoch number) on the far left
        axes[i, 0].set_ylabel(f"Epoch {epoch}", fontsize=12, rotation=90, labelpad=20)

        # Set column titles only for the first row
        if i == 0:
            axes[i, 0].set_title("Input")
            axes[i, 1].set_title("Target")
            axes[i, 2].set_title("Generated")
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(pad=0.5, h_pad=1.5)
    plt.savefig(output_path)
    plt.close(fig)



def denormalize(x):
    """
    Remove normalization from an image tensor.

    Args:
        x (torch.Tensor): Image tensor with shape (B, C, H, W).

    Returns:
        torch.Tensor: Image tensor with normalization removed.
    """
    # return x * 0.5 + 0.5  # Assuming normalization was done with mean=0.5 and std=0.5
    return x

def save_image(tensor: torch.Tensor, filename: str) -> None:
    """
    Save a tensor as an image file. Assumes input is float32 and already normalized.
    
    Args:
        tensor (torch.Tensor): image tensor in shapes like (H, W), (C, H, W),
                               (B, C, H, W), (H, W, C), or (B, H, W)
        filename (str): output image file path
    """
    logging.debug(f"Original tensor shape: {tensor.shape}")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    # Take first image if batched
    if tensor.dim() == 4:
        tensor = tensor[0]
        logging.debug(f"Reduced batch. New shape: {tensor.shape}")

    # Handle ambiguous 3D shapes
    elif tensor.dim() == 3:
        c0, c1, c2 = tensor.shape
        # Check if the first dimension is channels
        if c0 in (1, 3, 4):
            # Already (C, H, W)
            pass
        else:
            # Maybe (H, W, C)?
            if c2 in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
                logging.debug(f"Permuted (H, W, C) -> (C, H, W): {tensor.shape}")
            else:
                logging.warning(f"Ambiguous shape {tensor.shape}. Assuming (C, H, W).")

    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
        logging.debug(f"Added channel dimension: {tensor.shape}")
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    # Now tensor is (C, H, W)
    C, H, W = tensor.shape
    array = tensor.cpu().numpy()

    # Convert to (H, W) or (H, W, C)
    if C == 1:
        array = array.squeeze(0)             # -> (H, W)
        cmap = 'gray'
    else:
        array = array.transpose(1, 2, 0)     # -> (H, W, C)
        cmap = None

    plt.imsave(filename, array, cmap=cmap)
    
def save_float_image(image, filename: str):
    """
    Saves a float32 image (PyTorch tensor or NumPy array) to disk without altering values.

    Args:
        image (Union[torch.Tensor, np.ndarray]): 
            The image to save. Supported shapes:
            - PyTorch tensor: (H, W), (1, H, W), (C, H, W), (B, C, H, W)
            - NumPy array: (H, W), (H, W, C)
        filename (str): Output file path. Use a format that supports float32 (e.g., .tiff)
    
    Raises:
        ValueError: If the input shape is unsupported.
    """
    if isinstance(image, torch.Tensor):
        # Ensure tensor is on CPU and detached
        image = image.detach().cpu()

        # Remove batch dimension if present
        if image.dim() == 4:
            image = image.squeeze(0)  # [B, C, H, W] -> [C, H, W]

        # Rearrange dimensions
        if image.dim() == 3:
            image = image.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
        elif image.dim() == 2:
            pass  # [H, W] is fine
        else:
            raise ValueError(f"Unsupported tensor shape: {image.shape}")

        np_image = image.numpy().astype(np.float32)

    elif isinstance(image, np.ndarray):
        if image.ndim not in [2, 3]:
            raise ValueError(f"Unsupported ndarray shape: {image.shape}")
        np_image = image.astype(np.float32)
    else:
        raise TypeError(f"Input must be a torch.Tensor or np.ndarray, got {type(image)}")

    # Make sure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save using TIFF to preserve float32
    iio.imwrite(filename, np_image)

def count_parameters(model):
    """
    Count the number of learnable parameters in a model.

    Args:
        model (torch.nn.Module): The model to count parameters for.

    Returns:
        int: Total number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)