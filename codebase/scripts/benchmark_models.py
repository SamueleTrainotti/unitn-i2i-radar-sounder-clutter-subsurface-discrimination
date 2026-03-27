import sys
import scripting
import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict, List, Optional
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import yaml

from core import get_logger, get_config, Evaluator
from dataset import MapDataset
from models import Pix2Pix, CycleGAN

# Import functional metrics for per-sample calculation
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
from torchmetrics.functional import peak_signal_noise_ratio as psnr_func
import seaborn as sns

def _plot_metric_distributions(all_results_df: pd.DataFrame, output_dir: Path):
    """
    Generates boxplots/violin plots for SSIM, PSNR, LPIPS comparing all runs.
    """
    metrics = ['ssim', 'psnr', 'lpips']
    
    # Set style
    sns.set(style="whitegrid")
    
    for metric in metrics:
        if metric not in all_results_df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Create violin plot or boxplot
        # Fix: Assign x to hue and set legend=False to avoid FutureWarning
        sns.violinplot(x='run_id', y=metric, hue='run_id', data=all_results_df, inner='quartile', palette="muted", legend=False)
        sns.stripplot(x='run_id', y=metric, hue='run_id', data=all_results_df, size=2, palette="dark:.3", alpha=0.5, legend=False) # Add points
        
        plt.title(f'Distribution of {metric.upper()} Scores', fontsize=16)
        plt.xlabel('Run ID', fontsize=12)
        plt.ylabel(f'{metric.upper()}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = output_dir / f"{metric}_distribution.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
def _save_comparison_grid(sample_idx, real_A, real_B, model_outputs: Dict[str, torch.Tensor], output_dir: Path):
    """
    Saves a comparison grid: Input | Target | Model 1 | Model 2 ...
    """
    def to_numpy(t):
        if t.ndim == 4: t = t[0] # Take first from batch
        t = t.detach().cpu().squeeze()
        if t.ndim == 3: t = t.permute(1, 2, 0)
        t = (t + 1) / 2 # Assuming -1, 1 range from model
        t = torch.clamp(t, 0, 1)
        return t.numpy()

    real_A_np = to_numpy(real_A)
    real_B_np = to_numpy(real_B)
    
    num_models = len(model_outputs)
    num_cols = 2 + num_models
    
    # Increase base figure size slightly to prevent squishing
    fig, axs = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
    
    title_fontsize = 16
    
    # Input
    axs[0].imshow(real_A_np, cmap='gray')
    axs[0].set_title('Input (Simulated)', fontsize=title_fontsize)
    axs[0].axis('off')
    
    # Target
    axs[1].imshow(real_B_np, cmap='gray')
    axs[1].set_title('Target (Real)', fontsize=title_fontsize)
    axs[1].axis('off')
    
    # Models
    for i, (run_id, fake_B) in enumerate(model_outputs.items()):
        ax_idx = 2 + i
        fake_B_np = to_numpy(fake_B)
        axs[ax_idx].imshow(fake_B_np, cmap='gray')
        
        title = "Generated"
        if "cyclegan" in run_id.lower():
            title += " (CycleGan)"
        elif "pix2pix" in run_id.lower():
            title += " (Pix2Pix)"
        else:
            title += f" ({run_id[-8:]})"
            
        axs[ax_idx].set_title(title, fontsize=title_fontsize)
        axs[ax_idx].axis('off')
    
    # Increased padding to prevent cut offs
    plt.tight_layout(pad=2.0) 
    # bbox_inches='tight' asks matplotlib to expand the border exactly to fit everything without cropping
    plt.savefig(output_dir / f"comparison_sample_{sample_idx}.png", dpi=200, bbox_inches='tight')
    plt.close()


def load_model_from_run(run_id: str):
    logger = get_logger()
    config = get_config()
    logger.info(f"Loading model from run: {run_id}")
    
    base_dir = config.get("OUTPUT_DATA", {}).get("BASE_DIR")
    if not base_dir:
        logger.error("BASE_DIR not found in configuration.")
        return None

    run_dir = Path(base_dir) / run_id
    if not run_dir.is_dir():
        logger.error(f"Run directory not found: {run_dir}")
        return None

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        logger.error(f"Config file not found: {config_path}")
        return None

    with open(config_path) as f:
        run_config = yaml.safe_load(f)

    model_name = run_config.get("MODEL_NAME")
    if model_name == "pix2pix":
        model = Pix2Pix(training=False)
    elif model_name == "cyclegan":
        model = CycleGAN(training=False)
    else:
        logger.error(f"Unknown model name: {model_name}")
        return None
    
    model.config = run_config
    try:
        checkpoints_dir = run_dir / "checkpoints"
        model.load(str(checkpoints_dir))
        return model
    except Exception as e:
        logger.error(f"Failed to load model {run_id}: {e}")
        return None

def benchmark(run_ids: List[str], split: str = 'TEST', output_dir: str = None):
    logger = get_logger()
    config = get_config()
    
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path(config["OUTPUT_DATA"]["EVALUATION"]) / "benchmark"
    
    scripting.ensure_folder_exists(str(out_path))
    samples_dir = out_path / "samples"
    scripting.ensure_folder_exists(str(samples_dir))

    device = config['DEVICE']
    
    # 1. Load all models first
    loaded_models = {}
    for run_id in run_ids:
        model = load_model_from_run(run_id)
        if model:
            loaded_models[run_id] = model
            # Set to eval (already redundant if load_model_from_run does it, but good safety)
            for m in model.models.values():
                m.eval()
        else:
            logger.warning(f"Could not load model for run {run_id}, skipping.")
            
    if not loaded_models:
        logger.error("No models loaded. Exiting benchmark.")
        return

    # 2. Prepare dataset
    dataset_path = config["INPUT_DATA"].get(split)
    if not dataset_path:
        # Fallback for TEST split if not defined -> use VAL or error
        if split == 'TEST':
             logger.warning("TEST split not defined in config, falling back to VAL.")
             dataset_path = config["INPUT_DATA"].get('VAL')
             split = 'VAL'
        
        if not dataset_path:
            raise ValueError(f"Dataset split {split} not defined in config")
        
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = MapDataset(
        root_dir=dataset_path,
        temp_dir=config["OUTPUT_DATA"]["TEMP"],
        temp_dataset=config["OUTPUT_DATA"]["TEMP_DATASET"],
        SAVE_DATASET=config.get("SAVE_DATASET", False),
        LOAD_DATASET=config.get("LOAD_DATASET", False),
    )
    
    logger.info(f"Dataset loaded. Total patches available: {len(dataset)}")
    if hasattr(dataset, 'stats') and dataset.stats:
        logger.info(f"Raw original images in {split} split: {dataset.stats.get('valid_pairs', 'unknown')}")
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 3. Initialize evaluators and storage
    # One evaluator per model to track metrics independently (or reset between models if iterating, 
    # but here we iterate data outer loop, so we need separate instances or manual tracking)
    # Actually, Evaluator class has internal metrics that accumulate. We need one Evaluator per model
    # OR we manually compute and store per-sample metrics, then compute global means ourselves.
    # The latter is often cleaner for "Benchmark all at once" scripts.
    
    # Let's instantiate one Evaluator per model to use its metric accumulators (FID, etc.)
    evaluators = {run_id: Evaluator(device) for run_id in loaded_models}
    
    # Prepare storage for results
    per_sample_records = {run_id: [] for run_id in loaded_models}
    inference_times = {run_id: [] for run_id in loaded_models}
    
    # Reset metrics
    for ev in evaluators.values():
        ev.ssim_metric.reset()
        ev.psnr_metric.reset()
        ev.lpips_metric.reset()
        if ev.fid_metric:
            ev.fid_metric.reset()

    # 4. Iterate Dataset
    logger.info(f"Starting benchmark on {len(dataloader)} samples...")
    
    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(tqdm(dataloader, desc="Benchmarking")):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            if real_A.ndim == 3: real_A = real_A.unsqueeze(1)
            if real_B.ndim == 3: real_B = real_B.unsqueeze(1)
            
            # Helper for visualization
            model_outputs_for_vis = {}
            
            # Iterate Models
            for run_id, model in loaded_models.items():
                evaluator = evaluators[run_id]
                model_name = model.config.get("MODEL_NAME")
                
                # Select generator
                if model_name == 'cyclegan':
                    gen = model.get_generator("G_AB")
                else:
                    gen = model.get_generator()
                
                # Inference
                start_t = time.time()
                fake_B = gen(real_A)
                infer_t = time.time() - start_t
                inference_times[run_id].append(infer_t)
                
                # Normalization
                if fake_B.min() < 0:
                     fake_B_norm = (fake_B + 1) / 2
                     real_B_norm = (real_B + 1) / 2
                else:
                     fake_B_norm = fake_B
                     real_B_norm = real_B
                     
                fake_B_clamped = torch.clamp(fake_B_norm, 0, 1)
                real_B_clamped = torch.clamp(real_B_norm, 0, 1)
                
                fake_rgb = fake_B_clamped.repeat(1, 3, 1, 1)
                real_rgb = real_B_clamped.repeat(1, 3, 1, 1)
                
                # Metrics
                val_ssim = ssim_func(fake_B_norm, real_B_norm, data_range=1.0).item()
                val_psnr = psnr_func(fake_B_norm, real_B_norm, data_range=1.0).item()
                val_lpips = evaluator.lpips_metric(fake_rgb, real_rgb).item()
                
                per_sample_records[run_id].append({
                    'run_id': run_id,
                    'sample_idx': i,
                    'ssim': val_ssim,
                    'psnr': val_psnr,
                    'lpips': val_lpips,
                    'inference_time': infer_t
                })
                
                # Update Accumulators
                evaluator.ssim_metric.update(fake_B_norm, real_B_norm)
                evaluator.psnr_metric.update(fake_B_norm, real_B_norm)
                if evaluator.fid_metric:
                    # FID expects uint8
                    evaluator.fid_metric.update((real_rgb * 255).to(torch.uint8), real=True)
                    evaluator.fid_metric.update((fake_rgb * 255).to(torch.uint8), real=False)
                
                # Store for visualization
                if i < 5:
                    model_outputs_for_vis[run_id] = fake_B
            
            # Save Comparison Image
            if i < 5:
                _save_comparison_grid(i, real_A, real_B, model_outputs_for_vis, samples_dir)

    # 5. Finalize Results
    all_results = []
    summary_metrics = {}
    
    for run_id in loaded_models:
        evaluator = evaluators[run_id]
        
        # Compute Globals
        global_metrics = {
            'ssim_mean': evaluator.ssim_metric.compute().item(),
            'psnr_mean': evaluator.psnr_metric.compute().item(),
            'lpips_mean': evaluator.lpips_metric.compute().item(),
            'inference_time_mean': np.mean(inference_times[run_id])
        }
        
        if evaluator.fid_metric:
             try:
                global_metrics['fid'] = evaluator.fid_metric.compute().item()
             except Exception as e:
                 logger.warning(f"FID computation failed for {run_id}: {e}")
                 global_metrics['fid'] = -1.0
        
        summary_metrics[run_id] = global_metrics
        logger.info(f"Run {run_id} Metrics: {global_metrics}")
        
        # Save per-sample CSV
        df = pd.DataFrame(per_sample_records[run_id])
        csv_path = out_path / f"{run_id}_per_sample.csv"
        df.to_csv(csv_path, index=False)
        all_results.append(df)

    # Combine results and plot distributions
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        _plot_metric_distributions(combined_df, out_path)
        logger.info(f"Saved metric distribution plots to {out_path}")
        
    # Save Summary
    summary_path = out_path / "benchmark_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)
        
    logger.info(f"Benchmark Saved to {out_path}")

if __name__ == "__main__":
    # Create a wrapper to map args to benchmark function parameters
    def benchmark_wrapper(run_ids, split='TEST', out=None, **kwargs):
        print(f"DEBUG: benchmark_wrapper called with run_ids={run_ids}, split={split}, out={out}")
        if not run_ids:
             # This check is also done in the main logic but good to have
             print("Please provide run IDs using --run-ids")
             return
        benchmark(run_ids, split, out)

    def add_custom_args(parser):
        parser.add_argument("--split", type=str, default="TEST", help="Dataset split to use (TRAIN, VAL, TEST)")
        parser.add_argument("--out", type=str, default=None, help="Output directory")

    scripting.logged_main(
        description="Benchmark Models",
        main_fn=benchmark_wrapper,
        add_args_fn=add_custom_args,
        run_topic="benchmark"
    )
