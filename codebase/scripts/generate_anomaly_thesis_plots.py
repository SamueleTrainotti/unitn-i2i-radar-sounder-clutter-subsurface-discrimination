import scripting
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import yaml

from core import get_logger, get_config
from dataset import MapDataset
from models import CycleGAN, Pix2Pix
from core.anomaly_injector import RealisticAnomalyInjector


def _plot_metric_comparison(diff_l1, diff_l2, diff_ssim, output_dir, idx):
    """Generates a side-by-side heatmap comparison of L1, L2, and SSIM difference maps."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axs[0].imshow(diff_l1, cmap='hot', vmin=0, vmax=1)
    axs[0].set_title("L1 Absolute Difference", fontsize=16)
    axs[0].axis('off')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(diff_l2, cmap='hot', vmin=0, vmax=1)
    axs[1].set_title("L2 Squared Difference", fontsize=16)
    axs[1].axis('off')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(diff_ssim, cmap='hot', vmin=0, vmax=1)
    axs[2].set_title("1 - Structural Similarity (SSIM)", fontsize=16)
    axs[2].axis('off')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_dir / f"metric_comparison_{idx}.png", dpi=200, bbox_inches='tight')
    plt.close()


def _plot_sliding_window_effect(diff_map, window_sizes, output_dir, idx):
    """Shows how average pooling with increasing kernel sizes smooths the diff map."""
    fig, axs = plt.subplots(1, len(window_sizes) + 1, figsize=(6 * (len(window_sizes) + 1), 6))

    diff_tensor = torch.from_numpy(diff_map).unsqueeze(0).unsqueeze(0).float()

    axs[0].imshow(diff_map, cmap='hot', vmin=0, vmax=1)
    axs[0].set_title("Original Pixel Diff", fontsize=16)
    axs[0].axis('off')

    for i, w_size in enumerate(window_sizes):
        pool = nn.AvgPool2d(kernel_size=w_size, stride=1, padding=w_size // 2)
        smoothed_map = pool(diff_tensor).squeeze().numpy()

        axs[i + 1].imshow(smoothed_map, cmap='hot', vmin=0, vmax=1)
        axs[i + 1].set_title(f"Sliding Window ({w_size}x{w_size})", fontsize=16)
        axs[i + 1].axis('off')

    plt.tight_layout(pad=2.0)
    plt.savefig(output_dir / f"sliding_window_effect_{idx}.png", dpi=200, bbox_inches='tight')
    plt.close()


def _plot_threshold_effect(metrics_maps, thresholds, output_dir, idx):
    """
    Generates a grid plot (metrics x thresholds) to show how binary masks vary.
    Assumes metrics_maps are already normalized in [0, 1].
    """
    num_metrics = len(metrics_maps)
    num_thresh = len(thresholds)
    
    fig, axs = plt.subplots(num_metrics, num_thresh + 1, 
                            figsize=(4 * (num_thresh + 1), 4 * num_metrics))
    
    for row_idx, (m_name, diff_map) in enumerate(metrics_maps.items()):
        # Colonna 0: Mappa normalizzata
        im = axs[row_idx, 0].imshow(diff_map, cmap='hot', vmin=0, vmax=1)
        axs[row_idx, 0].set_title(f"Normalized {m_name}", fontsize=16)
        axs[row_idx, 0].axis('off')
        
        # Colonne successive: Maschere binarie
        for col_idx, thresh in enumerate(thresholds):
            binary_mask = (diff_map > thresh).astype(float)
            axs[row_idx, col_idx + 1].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
            axs[row_idx, col_idx + 1].set_title(f"Thresh {thresh}", fontsize=14)
            axs[row_idx, col_idx + 1].axis('off')
            
    plt.tight_layout(pad=2.0)
    plt.savefig(output_dir / f"threshold_effect_grid_{idx}.png", dpi=200, bbox_inches='tight')
    plt.close()


def generate_anomaly_plots(run_id: str, split: str = 'TEST', samples: int = 3, export_npz: bool = False, **kwargs):
    """Main entrypoint: loads a trained model, injects anomalies and generates thesis plots."""
    logger = get_logger()
    config = get_config()
    device = config['DEVICE']

    if not run_id:
        logger.error("No run ID provided.")
        return

    logger.info(f"Loading model from run: {run_id}")
    run_dir = Path(config["OUTPUT_DATA"]["BASE_DIR"]) / run_id
    with open(run_dir / "config.yaml") as f:
        run_config = yaml.safe_load(f)

    model_name = run_config.get("MODEL_NAME")
    if model_name == "pix2pix":
        model = Pix2Pix(training=False)
    elif model_name == "cyclegan":
        model = CycleGAN(training=False)
    else:
        logger.error(f"Unknown model: {model_name}")
        return

    model.config = run_config
    model.load(str(run_dir / "checkpoints"))
    generator = model.get_generator("G_AB") if model_name == 'cyclegan' else model.get_generator()
    generator.eval()

    logger.info(f"Loading dataset from {config['INPUT_DATA'][split]}")
    dataset = MapDataset(
        root_dir=config["INPUT_DATA"][split],
        temp_dir=config["OUTPUT_DATA"]["TEMP"],
        temp_dataset=config["OUTPUT_DATA"]["TEMP_DATASET"],
        LOAD_DATASET=config.get("LOAD_DATASET", False),
    )

    logger.info(f"Dataset loaded. Total patches: {len(dataset)}")
    if hasattr(dataset, 'stats') and dataset.stats:
        logger.info(f"Raw original images in {split} split: {dataset.stats.get('valid_pairs', 'unknown')}")

    anomaly_gen = RealisticAnomalyInjector(config)

    results_dir = Path(config["OUTPUT_DATA"]["EVALUATION"]) / "thesis_figures"
    scripting.ensure_folder_exists(str(results_dir))
    logger.info(f"Generating thesis anomaly plots in {results_dir}")

    # Use space-separated indices for a better distribution
    indices_to_plot = np.linspace(0, len(dataset) - 1, samples, dtype=int)
    
    exported_data = {"real": [], "sim": [], "reconstructed": []}

    for i in tqdm(indices_to_plot, desc="Generating Plot Examples"):
        sim, real = dataset[i]
        sim_batch = sim.unsqueeze(0).unsqueeze(0).to(device)
        real_batch = real.unsqueeze(0).unsqueeze(0).to(device)

        # Denormalize
        norm_type = config.get('NORMALIZATION_TYPE', 'range_minus_one_to_one')
        if norm_type == 'range_minus_one_to_one':
            real_denorm = (real_batch + 1) / 2
            sim_denorm = (sim_batch + 1) / 2
        else:
            real_denorm = real_batch
            sim_denorm = sim_batch

        real_denorm = torch.clamp(real_denorm, 0, 1)
        sim_denorm = torch.clamp(sim_denorm, 0, 1)

        # Inject anomaly
        real_injected, mask = anomaly_gen.inject_dipping_layer(real_denorm[0])
        real_injected_batch = real_injected.unsqueeze(0)

        # Reconstruct
        with torch.no_grad():
            reconstructed = generator(sim_batch)
            if norm_type == 'range_minus_one_to_one':
                reconstructed_denorm = (reconstructed + 1) / 2
            else:
                reconstructed_denorm = reconstructed
            reconstructed_denorm = torch.clamp(reconstructed_denorm, 0, 1)

        inj_np = real_injected_batch.squeeze().cpu().numpy()
        rec_np = reconstructed_denorm.squeeze().cpu().numpy()
        sim_np = sim_denorm.squeeze().cpu().numpy()

        if export_npz:
            exported_data["real"].append(inj_np)
            exported_data["sim"].append(sim_np)
            exported_data["reconstructed"].append(rec_np)

        # Difference maps
        diff_l1 = np.abs(inj_np - rec_np)
        diff_l2 = (inj_np - rec_np) ** 2

        # SSIM Map logic: ensure we use the full structural contrast
        data_range = inj_np.max() - inj_np.min()
        if data_range <= 0: data_range = 1.0
        _, ssim_map = ssim(inj_np, rec_np, full=True, data_range=data_range, win_size=7)
        diff_ssim = 1 - ssim_map 

        # --- NORMALIZZAZIONE 0-1 ---
        diff_l1_norm = normalize_diff_map(diff_l1)
        diff_l2_norm = normalize_diff_map(diff_l2)
        diff_ssim_norm = normalize_diff_map(diff_ssim)

        # Plots
        _plot_metric_comparison(diff_l1_norm, diff_l2_norm, diff_ssim_norm, results_dir, i)
        _plot_sliding_window_effect(diff_l1_norm, [3, 7, 15, 31], results_dir, i)
        
        # Grid plot with explicit normalization for SSIM
        metrics_to_thresh = {
            'L1': diff_l1_norm,
            'L2': diff_l2_norm,
            'SSIM': diff_ssim_norm
        }
        _plot_threshold_effect(metrics_to_thresh, [0.1, 0.3, 0.5], results_dir, i)

    if export_npz:
        export_path = results_dir / f"thesis_data_export_{run_id}.npz"
        np.savez_compressed(export_path, **{k: np.array(v) for k, v in exported_data.items()})
        logger.info(f"Exported raw data to: {export_path}")

    logger.info("Thesis plot generation complete.")

def normalize_diff_map(diff_map):
    """Applica il Min-Max scaling per portare la mappa esattamente nel range [0, 1]."""
    d_min, d_max = diff_map.min(), diff_map.max()
    if d_max > d_min:
        return (diff_map - d_min) / (d_max - d_min)
    return diff_map # Se l'immagine è piatta (d_max == d_min), la lascia invariata


if __name__ == "__main__":
    def _wrapper(run_ids=None, split='TEST', samples=3, export_npz=False, **kwargs):
        if not run_ids:
            print("ERROR: Please provide a run ID via --run-ids <run_id>")
            return
        run_id = run_ids[0] if isinstance(run_ids, list) else run_ids
        generate_anomaly_plots(run_id=run_id, split=split, samples=samples, export_npz=export_npz, **kwargs)

    def add_custom_args(parser):
        parser.add_argument("--split", type=str, default="TEST",
                            help="Dataset split (TRAIN, VAL, TEST)")
        parser.add_argument("--samples", type=int, default=3,
                            help="Number of sample plots to generate")
        parser.add_argument("--export-npz", action="store_true",
                            help="Export raw numpy data for local plotting")

    scripting.logged_main(
        description="Generate Thesis Visualizations for Anomaly Detection",
        main_fn=_wrapper,
        add_args_fn=add_custom_args,
        run_topic="thesis_plots"
    )
