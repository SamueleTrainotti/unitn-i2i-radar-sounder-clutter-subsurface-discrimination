import sys
import scripting
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import time
from tqdm import tqdm
from dataclasses import dataclass
from core import get_logger, get_config, load_config, Evaluator
from dataset import MapDataset
from models import Pix2Pix, CycleGAN
import matplotlib.pyplot as plt
import yaml
import torchvision

NUM_COMPARISON_SAMPLES = 5

class ModelComparator:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.device = self.config['DEVICE']
        self.evaluator = Evaluator(self.device)
        self.results = {}

    def load_model_from_run(self, run_id: str) -> Optional[Any]:
        self.logger.info(f"Loading model from run: {run_id}")
        base_dir = self.config.get("OUTPUT_DATA", {}).get("BASE_DIR")
        if not base_dir:
            self.logger.error("BASE_DIR not found in configuration.")
            return None

        run_dir = Path(base_dir) / run_id
        if not run_dir.is_dir():
            self.logger.error(f"Run directory not found: {run_dir}")
            return None

        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            self.logger.error(f"Config file not found in run directory: {config_path}")
            return None

        with open(config_path) as f:
            run_config = yaml.safe_load(f)

        model_name = run_config.get("MODEL_NAME")
        if model_name == "pix2pix":
            model = Pix2Pix()
        elif model_name == "cyclegan":
            model = CycleGAN()
        else:
            self.logger.error(f"Unknown model name in run {run_id}: {model_name}")
            return None
        
        # Set the model's config to the run-specific config before loading
        model.config = run_config

        try:
            # Construct the path to the checkpoint directory for this specific run
            checkpoints_dir = run_dir / "checkpoints"
            if not checkpoints_dir.is_dir():
                self.logger.error(f"Checkpoints directory not found in run {run_id}")
                raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

            # Let the model's load function handle finding the best/latest checkpoint within that directory
            model.load(str(checkpoints_dir))
            self.logger.info(f"Model '{model_name}' loaded successfully from run '{run_id}'.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model from run {run_id}: {e}", exc_info=True)
            return None

    def create_test_dataset(self) -> DataLoader:
        test_dataset = MapDataset(
            root_dir=self.config["INPUT_DATA"]["VAL"],
            temp_dir=self.config["OUTPUT_DATA"]["TEMP"],
            temp_dataset=self.config["OUTPUT_DATA"]["TEMP_DATASET"],
            SAVE_DATASET=self.config.get("SAVE_DATASET", False),
            LOAD_DATASET=self.config.get("LOAD_DATASET", False),
        )
        return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    def compare_models(self, run_ids: List[str], output_path: Optional[str] = None):
        self.logger.info("Starting comprehensive model comparison...")
        output_dir = Path(output_path) if output_path else Path(self.config["OUTPUT_DATA"]["EVALUATION"]) / "comparison_results"
        scripting.ensure_folder_exists(str(output_dir))

        models = {run_id: self.load_model_from_run(run_id) for run_id in run_ids}
        models = {k: v for k, v in models.items() if v is not None}

        if not models:
            raise ValueError("No models could be loaded for comparison.")

        results: Dict[str, Dict[str, float]] = {}

        for run_id, model in models.items():
            self.logger.info(f"Evaluating model from run: {run_id}")
            # Create a fresh data loader for each model to avoid exhaustion
            test_loader = self.create_test_dataset()
            
            model_name = model.config.get("MODEL_NAME")
            if model_name == 'pix2pix':
                metrics = self.evaluator.evaluate_model(model, test_loader)
                results[run_id] = metrics
            elif model_name == 'cyclegan':
                # For CycleGAN, we might want to evaluate both generators
                # This example evaluates G_AB (sim -> real)
                self.logger.info(f"Evaluating CycleGAN generator G_AB for run {run_id}")
                metrics_ab = self.evaluator.evaluate_model(model, test_loader) # Pass the whole model
                results[f"{run_id}_G_AB"] = metrics_ab

                # To evaluate G_BA (real -> sim), you would need to swap the datasets
                # This is a simplified example. For a full evaluation, you might need a more complex setup.
                self.logger.warning(f"CycleGAN G_BA evaluation is simplified. Ensure dataset is appropriate.")

        self._save_results(results, output_dir)
        self._log_results(results)
        self._generate_visualizations(results, output_dir)
        self._create_summary_table(results, output_dir)
        self._generate_sample_comparisons(models, self.create_test_dataset(), output_dir)

        self.logger.info(f"Comparison complete! Results saved to {output_dir}")
        return results

    def _log_results(self, results: Dict[str, Dict[str, float]]):
        self.logger.info("\n" + "="*20 + " Evaluation Results " + "="*20)
        for run_id, metrics in results.items():
            self.logger.info(f"\nResults for run: {run_id}")
            for metric_name, value in metrics.items():
                self.logger.info(f"  - {metric_name}: {value:.4f}")
        self.logger.info("\n" + "="*58)

    def _save_results(self, results: Dict[str, Dict[str, float]], output_dir: Path):
        results_path = output_dir / "comparison_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Saved detailed metrics to {results_path}")

    def _generate_visualizations(self, results: Dict[str, Dict[str, float]], output_dir: Path):
        self.logger.info("Generating metric visualizations...")
        
        # Determine all unique metric names across all runs
        metric_names = sorted(list(set(m for res in results.values() for m in res.keys())))
        
        for metric_name in metric_names:
            labels = []
            values = []
            for run_id, metrics in results.items():
                if metric_name in metrics:
                    labels.append(run_id)
                    values.append(metrics[metric_name])
            
            if not values:
                continue

            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, values)
            plt.ylabel(metric_name)
            plt.title(f"Comparison of {metric_name}")
            plt.xticks(rotation=45, ha="right")
            
            # Add data labels on top of bars
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom') # va: vertical alignment

            plt.tight_layout()
            plot_path = output_dir / f"{metric_name}_comparison.png"
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Saved {metric_name} comparison plot to {plot_path}")

    def _create_summary_table(self, results: Dict[str, Dict[str, float]], output_dir: Path):
        self.logger.info("Creating summary table visualization...")
        
        run_ids = list(results.keys())
        metric_names = sorted(list(set(m for res in results.values() for m in res.keys())))
        
        if not run_ids or not metric_names:
            self.logger.warning("No data available to create a summary table.")
            return

        cell_text = []
        for run_id in run_ids:
            row = [results[run_id].get(metric, 'N/A') for metric in metric_names]
            # Format numbers to 4 decimal places if they are indeed numbers
            formatted_row = [f'{x:.4f}' if isinstance(x, (int, float)) else x for x in row]
            cell_text.append(formatted_row)

        fig, ax = plt.subplots(figsize=(12, len(run_ids) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=cell_text, colLabels=metric_names, rowLabels=run_ids, cellLoc='center', loc='center')
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.title("Model Comparison Summary", fontsize=16)
        fig.tight_layout()

        table_path = output_dir / "comparison_summary_table.png"
        plt.savefig(table_path, dpi=300)
        plt.close()
        self.logger.info(f"Saved summary table to {table_path}")
        
    def _generate_sample_comparisons(self, models: Dict, test_loader: DataLoader, output_dir: Path):
        self.logger.info(f"Generating visual comparison for {NUM_COMPARISON_SAMPLES} samples...")
        
        # Ensure the output directory for images exists
        images_dir = output_dir / "image_comparisons"
        scripting.ensure_folder_exists(str(images_dir))

        # Get a few samples
        samples = list(zip(range(NUM_COMPARISON_SAMPLES), test_loader))

        def _plot_image(ax, tensor, title):
            img = tensor.cpu().detach()
            # Denormalize from [-1, 1] to [0, 1]
            img = img * 0.5 + 0.5
            
            if img.ndim == 3:
                img = img.permute(1, 2, 0)
            
            # If the image is [H, W, 1], squeeze it to [H, W] for grayscale display
            if img.ndim == 3 and img.shape[2] == 1:
                img = img.squeeze(2)

            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")

        for i, (x, y) in samples:
            x, y = x.to(self.device), y.to(self.device)
            
            # Determine the number of columns for the plot
            # 2 (input/target) + number of models
            num_cols = 2 + len(models)
            fig, axs = plt.subplots(1, num_cols, figsize=(num_cols * 4, 4))
            
            # Plot Input (Simulated) and Target (Real)
            _plot_image(axs[0], x[0], "Input (Simulated)")
            _plot_image(axs[1], y[0], "Target (Real)")

            col_idx = 2
            for run_id, model in models.items():
                
                with torch.no_grad():
                    # Handle different model types
                    if isinstance(model, Pix2Pix):
                        generator = model.get_generator()
                        generator.eval()
                        gen_output = generator(x.unsqueeze(1))
                    elif isinstance(model, CycleGAN):
                        # Assuming we are comparing sim-to-real generation
                        generator = model.get_generator() # Gets G_AB by default
                        generator.eval()
                        gen_output = generator(x.unsqueeze(1))
                    else:
                        # Fallback for unknown model types
                        gen_output = torch.zeros_like(x)

                _plot_image(axs[col_idx], gen_output[0], f"Run: {run_id[:8]}...")
                col_idx += 1
            
            plt.tight_layout()
            fig_path = images_dir / f"comparison_sample_{i}.png"
            plt.savefig(fig_path)
            plt.close(fig)
            self.logger.info(f"Saved comparison sample to {fig_path}")

def main(run_ids: List[str] = None):
    logger = get_logger()
    if not run_ids:
        logger.error("No run IDs provided for comparison. Use the --runs argument.")
        return

    logger.info("="*50)
    logger.info("Starting model comparison...")
    comparator = ModelComparator()
    comparator.compare_models(run_ids=run_ids)
    logger.info("Model comparison completed successfully.")
    logger.info("="*50)

if __name__ == "__main__":
    scripting.logged_main(
        "Model Comparison",
        main,
    )