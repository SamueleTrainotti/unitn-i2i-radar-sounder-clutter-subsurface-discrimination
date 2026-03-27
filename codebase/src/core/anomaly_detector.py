import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import lpips
import matplotlib.pyplot as plt

from dataset.processing import apply_median_filter
from core.config import get_config

def pca_denoise(radargram, num_components=2):
    """
    Denoise a radargram using PCA (low-rank approximation) implemented via SVD.

    This function treats `radargram` as a 2D array with shape (n_samples, n_traces),
    i.e. rows are time/depth samples and columns are traces (spatial positions).

    The algorithm is:
    - center each row by subtracting the mean across traces (keep mean per sample)
    - compute a truncated SVD of the centered matrix
    - reconstruct the matrix using the top `num_components` singular values/vectors
    - add the mean back

    Using SVD is numerically stable and avoids forming the (potentially large)
    covariance matrix explicitly. The previous implementation attempted an
    eigen-decomposition of the covariance but had shape/ordering ambiguities
    that make it unreliable.

    Args:
        radargram (np.ndarray): 2D array (n_samples, n_traces).
        num_components (int): Number of principal components to keep.

    Returns:
        np.ndarray: Reconstructed (denoised) radargram of same shape.
    """
    if radargram is None or radargram.ndim != 2:
        raise ValueError("radargram must be a 2D numpy array with shape (n_samples, n_traces)")

    # center across traces for each time sample
    mean_trace = np.mean(radargram, axis=1, keepdims=True)
    X = radargram - mean_trace

    # Compute SVD: X = U @ S @ Vt
    # truncated reconstruction: X_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fall back to eigh on covariance (smaller dimension) if SVD fails
        cov = np.cov(X, rowvar=True)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:, idx]
        k = min(num_components, eigvecs.shape[1])
        projected = X.T @ eigvecs[:, :k]
        reconstructed = (projected @ eigvecs[:, :k].T).T
        return reconstructed + mean_trace

    k = max(1, min(num_components, S.size))
    # efficient truncated reconstruction
    Uk = U[:, :k]
    Sk = S[:k]
    Vtk = Vt[:k, :]
    reconstructed = (Uk * Sk) @ Vtk

    return reconstructed + mean_trace

def subtract_average_trace(radargram):
    """
    Subtracts the average trace (A-scan) from the radargram.

    Args:
        radargram (numpy.ndarray): The input radargram.

    Returns:
        numpy.ndarray: The radargram with the average trace subtracted.
    """
    if radargram is None or radargram.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")
    
    average_trace = np.mean(radargram, axis=1, keepdims=True)
    return radargram - average_trace

def apply_tvg(radargram, gain_type='linear', gain_factor=1.0):
    """
    Applies Time-Varying Gain (TVG) to the radargram.

    Args:
        radargram (numpy.ndarray): The input radargram.
        gain_type (str): The type of gain function ('linear', 'exponential', 'power').
        gain_factor (float): A factor to control the strength of the gain.

    Returns:
        numpy.ndarray: The radargram with TVG applied.
    """
    if radargram is None or radargram.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    n_samples = radargram.shape[0]
    t = np.arange(n_samples)

    if gain_type == 'linear':
        gain = 1 + t * gain_factor / n_samples
    elif gain_type == 'exponential':
        gain = np.exp(t * gain_factor / n_samples)
    elif gain_type == 'power':
        gain = (1 + t / n_samples) ** gain_factor
    else:
        raise ValueError(f"Unknown gain type: {gain_type}")

    return radargram * gain[:, np.newaxis]

class AnomalyDetector:
    """
    A class to detect anomalies in radargrams using a trained generator model.
    """
    def __init__(self, generator_model, device):
        """
        Initializes the AnomalyDetector.

        Args:
            generator_model (nn.Module): The trained generator model.
            device (str): The device to run the model on.
        """
        self.generator_model = generator_model
        self.device = device
        self.generator_model.to(self.device)
        self.generator_model.eval()
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
        self.config = get_config()

    def _denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalizes a tensor to the [0, 1] range for visualization and metrics.

        This method reads the `NORMALIZATION_TYPE` from the configuration
        and applies the appropriate inverse transformation.
        - 'range_minus_one_to_one': Assumes input is in [-1, 1] and scales to [0, 1].
        - 'range_zero_to_one': Assumes input is already in [0, 1] and does nothing.
        
        Defaults to 'range_minus_one_to_one' if the setting is not found.

        Args:
            tensor (torch.Tensor): The input tensor to denormalize.

        Returns:
            torch.Tensor: The denormalized tensor in the [0, 1] range.
        """
        norm_type = self.config.get('NORMALIZATION_TYPE', 'range_minus_one_to_one')
        
        if norm_type == 'range_minus_one_to_one':
            return (tensor + 1) / 2
        elif norm_type == 'range_zero_to_one':
            return tensor
        else:
            # Fallback for unknown normalization type, defaults to the most common case.
            return (tensor + 1) / 2

    def _plot_image(self, ax, image, title, cmap=None, show_colorbar=False, is_patch_based=False):
        """Helper function to plot a single image."""
        image_np = image.squeeze().cpu().numpy()
        interpolation = 'nearest' if is_patch_based else 'none'
        im = ax.imshow(image_np, cmap=cmap, interpolation=interpolation)
        if self.config['PLOTTING']['SHOW_TITLES']:
            ax.set_title(title)
        ax.axis('off')
        if show_colorbar:
            plt.colorbar(im, ax=ax)

    def _processing_pipeline(self, real_img, reconstructed_img):
        """
        Applies a series of processing steps to the real and reconstructed images.
        """
        anomaly_config = self.config.get('ANOMALY_DETECTION', {})
        # 1. Z-score normalization
        z_score_config = anomaly_config.get('Z_SCORE_NORMALIZATION', {})
        if z_score_config.get('ENABLED', False):
            # Normalize real image
            mean_real = torch.mean(real_img)
            std_real = torch.std(real_img)
            real_img = (real_img - mean_real) / (std_real + 1e-8)

            # Normalize reconstructed image
            mean_reconstructed = torch.mean(reconstructed_img)
            std_reconstructed = torch.std(reconstructed_img)
            reconstructed_img = (reconstructed_img - mean_reconstructed) / (std_reconstructed + 1e-8)

        # 2. Median filtering and Gaussian blur
        median_config = anomaly_config.get('MEDIAN_FILTER', {})
        if median_config.get('ENABLED', False):
            window_size = median_config.get('WINDOW_SIZE', 3)
            real_img = apply_median_filter(real_img, window_size)
            reconstructed_img = apply_median_filter(reconstructed_img, window_size)

        blur_config = anomaly_config.get('GAUSSIAN_BLUR', {})
        if blur_config.get('ENABLED', False):
            sigma = blur_config.get('SIGMA', 1.0)
            kernel_size = 2 * int(3.0 * sigma + 0.5) + 1
            blurrer = GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
            real_img = blurrer(real_img)
            reconstructed_img = blurrer(reconstructed_img)

        # 3. PCA and other background removal methods
        br_config = anomaly_config.get('BACKGROUND_REMOVAL', {})
        br_method = br_config.get('METHOD', 'none')

        if br_method != 'none':
            real_np = real_img.squeeze().cpu().numpy()
            reconstructed_np = reconstructed_img.squeeze().cpu().numpy()

            if br_method == 'pca':
                num_components = br_config.get('PCA_COMPONENTS', 2)
                real_processed_np = pca_denoise(real_np, num_components=num_components)
                reconstructed_processed_np = pca_denoise(reconstructed_np, num_components=num_components)
            elif br_method == 'avg_trace':
                real_processed_np = subtract_average_trace(real_np)
                reconstructed_processed_np = subtract_average_trace(reconstructed_np)
            else:
                real_processed_np = real_np
                reconstructed_processed_np = reconstructed_np
            
            real_img = torch.from_numpy(real_processed_np).unsqueeze(0).unsqueeze(0).to(self.device)
            reconstructed_img = torch.from_numpy(reconstructed_processed_np).unsqueeze(0).unsqueeze(0).to(self.device)

        return real_img, reconstructed_img
    
    def _calculate_diff_maps(self, real_processed, reconstructed_processed, metrics, ssim_window_size=7):
        """Calculates pixel-level difference maps for a given set of metrics."""
        diff_maps = {}

        if "L1" in metrics:
            diff_maps["L1"] = torch.abs(real_processed - reconstructed_processed)
        if "L2" in metrics:
            diff_maps["L2"] = (real_processed - reconstructed_processed) ** 2

        if "SSIM" in metrics:
            real_np = real_processed.squeeze().cpu().numpy()
            reconstructed_np = reconstructed_processed.squeeze().cpu().numpy()
            data_range = real_np.max() - real_np.min()
            win_size = min(ssim_window_size, min(real_np.shape[0], real_np.shape[1]))
            if win_size % 2 == 0: win_size -= 1
            
            _ssim_value, ssim_diff_np = ssim(real_np, reconstructed_np, full=True, data_range=data_range, win_size=win_size)
            diff_maps["SSIM"] = torch.from_numpy(1 - ssim_diff_np).unsqueeze(0).unsqueeze(0).to(self.device)

        return diff_maps

    def _calculate_patch_diffs(self, diff_maps, patch_size):
        """Calculates patch-based difference scores using average pooling."""
        patch_diffs = {}
        pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size, padding=0)
        for metric, diff_map in diff_maps.items():
            patch_diffs[metric] = pool(diff_map)
        return patch_diffs

    def _calculate_sliding_window_diffs(self, diff_maps, window_size):
        """Calculates sliding window difference scores using average pooling."""
        processed_diffs = {}
        pool = nn.AvgPool2d(kernel_size=window_size, stride=1, padding=window_size // 2)
        for metric, diff_map in diff_maps.items():
            if metric in ["L1", "L2"]:
                processed_diffs[metric] = pool(diff_map)
            else:
                processed_diffs[metric] = diff_map
        return processed_diffs

    def _calculate_lpips(self, real_norm, reconstructed_norm, metrics):
        """Calculates the LPIPS value if requested."""
        if "LPIPS" in metrics:
            return self.lpips_loss(real_norm.float(), reconstructed_norm.float()).item()
        return None

    def save_comparison_plot(self, index, output_dir, simulated, real, reconstructed,
                             diff_maps, processed_diffs, anomaly_masks, lpips_value, is_patch_based=False):
        """Saves a grid of images using Matplotlib for a single sample."""
        num_metrics = len(diff_maps)
        num_cols = max(3, num_metrics)
        fig, axs = plt.subplots(3, num_cols, figsize=(5 * num_cols, 15), squeeze=False)

        self._plot_image(axs[0, 0], simulated, "Simulated Input")
        self._plot_image(axs[0, 1], real, "Real Radargram (Ground Truth)")
        self._plot_image(axs[0, 2], reconstructed, "Reconstructed Radargram")
        for j in range(3, num_cols):
            axs[0, j].axis('off')

        method = self.config.get('ANOMALY_DETECTION', {}).get('METHOD', 'pixel-level')
        for i, (metric, diff_map) in enumerate(sorted(diff_maps.items())):
            if i < num_cols:
                title = f"{metric} Difference"
                if metric == "SSIM":
                    ssim_score = 1 - diff_map.mean().cpu().item()
                    title += f"\n(Score: {ssim_score:.4f})"

                # For patch-based, we plot the processed diff map
                if method == 'patch-based':
                    plot_map = processed_diffs[metric]
                    self._plot_image(axs[1, i], plot_map, title, cmap='hot', show_colorbar=True, is_patch_based=True)
                elif method == 'sliding-window':
                    plot_map = processed_diffs[metric]
                    self._plot_image(axs[1, i], plot_map, title, cmap='hot', show_colorbar=True, is_patch_based=False)
                elif method == 'whole-image':
                    # For whole-image, we display the mean difference value
                    mean_diff = processed_diffs[metric].item()
                    axs[1, i].text(0.5, 0.5, f'Mean diff:\n{mean_diff:.4f}', 
                                 horizontalalignment='center', verticalalignment='center', 
                                 transform=axs[1, i].transAxes, fontsize=12)
                    axs[1, i].set_title(title)
                    axs[1, i].axis('off')
                else: # pixel-level
                    self._plot_image(axs[1, i], diff_map, title, cmap='hot', show_colorbar=True)
        for j in range(num_metrics, num_cols):
            axs[1, j].axis('off')

        for i, (metric, mask) in enumerate(sorted(anomaly_masks.items())):
            if i < num_cols:
                title = f"{metric} Anomaly Mask\n(Threshold: {self.config['THRESHOLD']})"
                self._plot_image(axs[2, i], mask, title, cmap='gray', is_patch_based=is_patch_based)
        for j in range(num_metrics, num_cols):
            axs[2, j].axis('off')

        if self.config['PLOTTING']['SHOW_TITLES']:
            lpips_text = f"\nLPIPS Score: {lpips_value:.4f}" if lpips_value is not None else ""
            fig.suptitle(f"Anomaly Detection Analysis - Image {index}{lpips_text}", fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"{index:04d}_matplotlib_comparison.png"))
        plt.close(fig)

    def _calculate_std_map(self, image, window_size):
        """Calculates a standard deviation map using a sliding window."""
        padding = window_size // 2
        image_2d = image.squeeze()
        image_padded = torch.nn.functional.pad(image_2d.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect').squeeze()
        
        patches = image_padded.unfold(0, window_size, 1).unfold(1, window_size, 1)
        std_dev = patches.contiguous().view(patches.size(0), patches.size(1), -1).std(dim=-1)
        
        return std_dev
    
    def save_extended_comparison_plot(self, index, output_dir, real_raw, reconstructed_raw, real_tvg, reconstructed_tvg, real_processed, reconstructed_processed, real_norm_z, reconstructed_norm_z, diff_map):
        """Saves a more detailed comparison plot for a single sample."""
        fig, axs = plt.subplots(5, 3, figsize=(20, 20), squeeze=False)

        # --- Row 1: Image values with colorbar ---
        self._plot_image(axs[0, 0], torch.from_numpy(real_raw), "Real (Raw)", cmap='viridis', show_colorbar=True)
        self._plot_image(axs[0, 1], torch.from_numpy(reconstructed_raw), "Reconstructed (Raw)", cmap='viridis', show_colorbar=True)
        if diff_map is not None:
            self._plot_image(axs[0, 2], diff_map, "L1 Difference", cmap='hot', show_colorbar=True)
        else:
            axs[0, 2].axis('off')


        # --- Row 2: A-scan plots (processed) ---
        real_processed_np = real_processed.squeeze().cpu().numpy()
        reconstructed_processed_np = reconstructed_processed.squeeze().cpu().numpy()
        width = real_processed_np.shape[1]
        locations = [width // 4, width // 2, 3 * width // 4]
        titles = ["A-Scan (Processed, 1/4)", "A-Scan (Processed, 1/2)", "A-Scan (Processed, 3/4)"]

        for i, (loc, title) in enumerate(zip(locations, titles)):
            axs[1, i].plot(real_processed_np[:, loc], label='Real')
            axs[1, i].plot(reconstructed_processed_np[:, loc], label='Reconstructed', linestyle='--')
            axs[1, i].set_title(title)
            axs[1, i].set_xlabel("Depth (pixels)")
            axs[1, i].set_ylabel("Amplitude")
            axs[1, i].legend()
            axs[1, i].grid(True)

        # --- Row 3: A-scan plots (Z-score normalized) ---
        if real_norm_z is not None and reconstructed_norm_z is not None:
            real_norm_z_np = real_norm_z.squeeze().cpu().numpy()
            reconstructed_norm_z_np = reconstructed_norm_z.squeeze().cpu().numpy()
            width = real_norm_z_np.shape[1]
            locations = [width // 4, width // 2, 3 * width // 4]
            titles = ["A-Scan (Z-Score, 1/4)", "A-Scan (Z-Score, 1/2)", "A-Scan (Z-Score, 3/4)"]

            for i, (loc, title) in enumerate(zip(locations, titles)):
                axs[2, i].plot(real_norm_z_np[:, loc], label='Real (Z-Score)')
                axs[2, i].plot(reconstructed_norm_z_np[:, loc], label='Reconstructed (Z-Score)', linestyle='--')
                axs[2, i].set_title(title)
                axs[2, i].set_xlabel("Depth (pixels)")
                axs[2, i].set_ylabel("Amplitude")
                axs[2, i].legend()
                axs[2, i].grid(True)
        else:
            for i in range(3):
                axs[2, i].text(0.5, 0.5, 'Z-Score Normalization Not Applied', 
                             horizontalalignment='center', verticalalignment='center', 
                             transform=axs[2, i].transAxes, fontsize=10)
                axs[2, i].axis('off')

        # --- Row 4: Standard Deviation Maps ---
        std_window_size = self.config.get('ANOMALY_DETECTION', {}).get('STD_WINDOW_SIZE', 11)
        
        std_real = self._calculate_std_map(real_processed.squeeze(), std_window_size)
        std_reconstructed = self._calculate_std_map(reconstructed_processed.squeeze(), std_window_size)
        if diff_map is not None:
            std_diff = self._calculate_std_map(diff_map, std_window_size)
            self._plot_image(axs[3, 2], std_diff, "Std Dev - Difference", cmap='viridis', show_colorbar=True)

        self._plot_image(axs[3, 0], std_real, "Std Dev - Real (Processed)", cmap='viridis', show_colorbar=True)
        self._plot_image(axs[3, 1], std_reconstructed, "Std Dev - Reconstructed (Processed)", cmap='viridis', show_colorbar=True)
        
        # --- Row 5: TVG effect on A-scan ---
        loc = width // 2 # Middle A-scan
        axs[4, 0].plot(real_raw[:, loc], label='Raw')
        if real_tvg is not None:
            axs[4, 0].plot(real_tvg[:, loc], label='TVG Applied', linestyle='--')
        else:
             axs[4, 0].text(0.5, 0.5, 'TVG Not Applied', 
                             horizontalalignment='center', verticalalignment='center', 
                             transform=axs[4, 0].transAxes, fontsize=10)

        axs[4, 0].set_title("TVG Effect on Real A-Scan (middle)")
        axs[4, 0].set_xlabel("Depth (pixels)")
        axs[4, 0].set_ylabel("Amplitude")
        axs[4, 0].legend()
        axs[4, 0].grid(True)

        axs[4, 1].axis('off') # Hide unused subplot
        axs[4, 2].axis('off') # Hide unused subplot
        
        if self.config['PLOTTING']['SHOW_TITLES']:
            fig.suptitle(f"Extended Analysis - Image {index}", fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"{index:04d}_extended_comparison.png"))
        plt.close(fig)

    def _generate_summary_plot(self, output_dir, all_diffs, is_patch_based=False):
        """Generates and saves a summary plot of average difference maps."""
        num_metrics = len(all_diffs)
        if num_metrics == 0: return

        fig, axs = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 6), squeeze=False)

        for i, (metric, diff_list) in enumerate(sorted(all_diffs.items())):
            avg_diff = torch.mean(torch.cat(diff_list), dim=0, keepdim=True)
            self._plot_image(axs[0, i], avg_diff, f"Average {metric} Difference", cmap='hot', show_colorbar=True, is_patch_based=is_patch_based)

        if self.config['PLOTTING']['SHOW_TITLES']:
            fig.suptitle("Average Anomaly Maps Across All Test Images", fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, "summary_average_anomalies.png"))
        plt.close(fig)

    def detect_anomalies(self, dataset, output_dir, threshold=0.5, sample_index=None):
        os.makedirs(output_dir, exist_ok=True)
        
        ad_config = self.config.get('ANOMALY_DETECTION', {})
        metrics = ad_config.get('METRICS', ['L1', 'L2', 'SSIM', 'LPIPS'])
        method = ad_config.get('METHOD', 'pixel-level')
        patch_size = ad_config.get('PATCH_SIZE', 64)
        ssim_window_size = ad_config.get('SSIM_WINDOW_SIZE', 7)
        l1_l2_window_size = ad_config.get('L1_L2_WINDOW_SIZE', 11)
        is_patch_based = (method == 'patch-based')

        summary_diffs = {metric: [] for metric in metrics if metric not in ['LPIPS']}
        all_lpips = [] if 'LPIPS' in metrics else None

        for i, (simulated, real) in enumerate(dataset):
            current_index = sample_index if sample_index is not None else i
            
            simulated = simulated.unsqueeze(0).unsqueeze(0).to(self.device)
            real = real.unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                reconstructed = self.generator_model(simulated)

            real_processed, reconstructed_processed = self._processing_pipeline(real, reconstructed)

            simulated_norm = self._denormalize_tensor(simulated)
            real_norm = self._denormalize_tensor(real_processed)
            reconstructed_norm = self._denormalize_tensor(reconstructed_processed)
            
            diff_maps = self._calculate_diff_maps(real_norm, reconstructed_norm, metrics, ssim_window_size)
            lpips_value = self._calculate_lpips(real_norm, reconstructed_norm, metrics)
            if lpips_value is not None: all_lpips.append(lpips_value)

            if method == 'whole-image':
                processed_diffs = {m: torch.mean(d) for m, d in diff_maps.items()}
            elif method == 'patch-based':
                processed_diffs = self._calculate_patch_diffs(diff_maps, patch_size)
            elif method == 'pixel-level':
                processed_diffs = diff_maps
            elif method == 'sliding-window':
                processed_diffs = self._calculate_sliding_window_diffs(diff_maps, l1_l2_window_size)
            else:
                raise ValueError(f"Unknown anomaly detection method: {method}")

            anomaly_masks = {m: (d > threshold).float() for m, d in processed_diffs.items()}
            
            for metric, p_diff in processed_diffs.items():
                summary_diffs[metric].append(p_diff.cpu())

            log_message = f"Processed image {current_index}"
            if "SSIM" in diff_maps:
                log_message += f", SSIM: {1 - diff_maps['SSIM'].mean().cpu().item():.4f}"
            if lpips_value is not None:
                log_message += f", LPIPS: {lpips_value:.4f}"
            print(log_message)

            if self.config['PLOTTING']['USE_MATPLOTLIB']:
                self.save_comparison_plot(current_index, output_dir, simulated_norm, real_norm, reconstructed_norm,
                                          diff_maps, processed_diffs, anomaly_masks, lpips_value, is_patch_based)
                if self.config['PLOTTING'].get('GENERATE_EXTENDED_PLOT', False):
                    self.save_extended_comparison_plot(
                        current_index, output_dir, 
                        real.squeeze().cpu().numpy(), reconstructed.squeeze().cpu().numpy(),
                        None, None, # tvg
                        real_processed, reconstructed_processed,
                        None, None, # z_score
                        diff_maps.get("L1")
                    )
            else:
                self.save_image_grid(current_index, output_dir, simulated_norm, real_norm, reconstructed_norm,
                                     diff_maps, anomaly_masks, lpips_value, is_patch_based)

        if self.config['PLOTTING']['GENERATE_SUMMARY']:
            self._generate_summary_plot(output_dir, summary_diffs, is_patch_based)

    def save_image_grid(self, index, output_dir, simulated, real, reconstructed,
                        diff_maps, anomaly_masks, lpips_value, is_patch_based=False):
        grid_images = [simulated, real, reconstructed]
        for metric in sorted(diff_maps.keys()):
            grid_images.append(diff_maps[metric])
        for metric in sorted(anomaly_masks.keys()):
            if is_patch_based:
                mask = nn.functional.interpolate(anomaly_masks[metric], size=simulated.shape[2:], mode='nearest')
                grid_images.append(mask)
            else:
                grid_images.append(anomaly_masks[metric])

        num_cols = max(3, len(diff_maps))
        while len(grid_images) % num_cols != 0:
            grid_images.append(torch.zeros_like(simulated))
        
        grid = torch.cat(grid_images, dim=0)
        save_image(grid, os.path.join(output_dir, f"{index:04d}_comparison_grid.png"), nrow=num_cols)

        save_image(simulated, os.path.join(output_dir, f"{index:04d}_01_simulated.png"))
        save_image(real, os.path.join(output_dir, f"{index:04d}_02_real.png"))
        save_image(reconstructed, os.path.join(output_dir, f"{index:04d}_03_reconstructed.png"))
        
        img_counter = 4
        for metric, diff_map in sorted(diff_maps.items()):
            save_image(diff_map, os.path.join(output_dir, f"{index:04d}_{img_counter:02d}_{metric}_diff.png"))
            img_counter += 1

        if lpips_value is not None:
            with open(os.path.join(output_dir, f"{index:04d}_{img_counter:02d}_lpips.txt"), 'w') as f:
                f.write(str(lpips_value))
            img_counter += 1

        for metric, mask in sorted(anomaly_masks.items()):
            save_image(mask, os.path.join(output_dir, f"{index:04d}_{img_counter:02d}_{metric}_mask.png"))
            img_counter += 1