import scripting
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import torch.nn.functional as F
import random
import yaml
import sys

from core import get_logger, get_config
from dataset import MapDataset
from models import CycleGAN, Pix2Pix
from core.anomaly_detector import AnomalyDetector

from core.anomaly_injector import RealisticAnomalyInjector

def evaluate_anomalies(run_ids: list, split: str = 'TEST', num_samples: int = 100, **kwargs):
    logger = get_logger()
    config = get_config()
    device = config['DEVICE']
    
    if not run_ids:
        logger.error("No run IDs provided.")
        return

    # For now, let's just evaluate the first run ID if multiple are provided, or iterate.
    # Iterating is better.
    for run_id in run_ids:
        logger.info(f"Evaluating Run: {run_id}")
    
        # Load Model
        run_dir = Path(config["OUTPUT_DATA"]["BASE_DIR"]) / run_id
        with open(run_dir / "config.yaml") as f:
            run_config = yaml.safe_load(f)
            
        model_name = run_config.get("MODEL_NAME")
        if model_name == "pix2pix":
            model = Pix2Pix(training=False)
        elif model_name == "cyclegan":
            model = CycleGAN(training=False)
        
        model.config = run_config
        model.load(str(run_dir / "checkpoints"))
        model.models['G_AB'].eval() if model_name == 'cyclegan' else model.generator.eval()
        generator = model.get_generator("G_AB") if model_name == 'cyclegan' else model.get_generator()

        # Load Dataset
        dataset = MapDataset(
            root_dir=config["INPUT_DATA"][split],
            temp_dir=config["OUTPUT_DATA"]["TEMP"],
            temp_dataset=config["OUTPUT_DATA"]["TEMP_DATASET"],
            LOAD_DATASET=config.get("LOAD_DATASET", False),
        )
        
        logger.info(f"Dataset loaded. Total patches available: {len(dataset)}")
        if hasattr(dataset, 'stats') and dataset.stats:
            logger.info(f"Raw original images in {split} split: {dataset.stats.get('valid_pairs', 'unknown')}")
        
        # Anomaly Generator
        anomaly_gen = RealisticAnomalyInjector(config)
        detector = AnomalyDetector(generator, device)
        
        aucs = []
        ap_scores = []
        all_y_true = []
        all_y_scores = []
        
        results_dir = Path(config["OUTPUT_DATA"]["EVALUATION"]) / f"evaluation"
        scripting.ensure_folder_exists(str(results_dir))
        
        # Evaluation Loop
        if num_samples is None or num_samples == -1:
             sample_indices = range(len(dataset))
        else:
             sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        for i in tqdm(sample_indices, desc="Evaluating Anomalies"):
            sim, real = dataset[i]
            sim = sim.unsqueeze(0).unsqueeze(0).to(device)
            real = real.unsqueeze(0).unsqueeze(0).to(device)
            
            # 1. Inject Anomaly into Real
            # Use the new injector. real is [1, 1, H, W]
            real_injected, mask = anomaly_gen.inject_dipping_layer(real[0])
            # Restore batch dim for generated mask if needed, but inject_dipping_layer returns [1, H, W] and [1, 1, H, W]
            # Wait, inject_dipping_layer returns injected [1, C, H, W] and mask [1, 1, H, W]?
            # Let's check the new code: returns injected_image.unsqueeze(0) where input was image_tensor [C, H, W]
            # So output is [1, C, H, W]. Mask is [1, 1, H, W].
            
            # real is [B=1, C=1, H, W]
            # inject_dipping_layer expects [C, H, W]
            real_injected, mask = anomaly_gen.inject_dipping_layer(real[0])
            real_injected = real_injected.unsqueeze(0) # Add batch dim back -> [1, 1, C, H, W]? No.
            # inject_dipping_layer returns [1, C, H, W]. So we just need to use it.
            # Actually, let's verify dimensions.
            # Input real[0] is [C, H, W].
            # inject returns injected_image.unsqueeze(0) -> [1, C, H, W]. This is batch size 1.
            # Perfect.
            
            # Ensure dims match
            if real_injected.dim() == 3: real_injected = real_injected.unsqueeze(0)
            
            # 2. Reconstruct from Sim (The generator shouldn't know about the anomaly in Real)
            with torch.no_grad():
                reconstructed = generator(sim)
                
            # 3. Compute Difference (L1 for simplicity, or grab from AnomalyDetector methods)
            # We can use AnomalyDetector._calculate_diff_maps, but it acts on denormalized data
            
            # Quick manual pipeline for control
            diff = torch.abs(real_injected - reconstructed)
            
            # Normalize diff for scoring (optional)
            diff_map = diff.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            
            # 4. Compute Metrics
            # Flatten
            y_true = mask_np.flatten()
            y_score = diff_map.flatten()
            
            # Only compute if there is an anomaly (mask has positive values)
            if y_true.sum() > 0:
                all_y_true.extend(y_true)
                all_y_scores.extend(y_score)
                
                try:
                    auc_score = roc_auc_score(y_true, y_score)
                    precision, recall, _ = precision_recall_curve(y_true, y_score)
                    ap_score = auc(recall, precision)
                    
                    aucs.append(auc_score)
                    ap_scores.append(ap_score)
                    
                    # Save first few for visualization
                    if len(aucs) <= 5:
                        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
                        axs[0].imshow(sim.squeeze().cpu(), cmap='gray'); axs[0].set_title("Simulated")
                        axs[1].imshow(real.squeeze().cpu(), cmap='gray'); axs[1].set_title("Original Real")
                        axs[2].imshow(real_injected.squeeze().cpu(), cmap='gray'); axs[2].set_title("Injected Real")
                        axs[3].imshow(reconstructed.squeeze().cpu(), cmap='gray'); axs[3].set_title("Reconstructed")
                        axs[4].imshow(diff_map, cmap='hot'); axs[4].set_title(f"Diff (AUC: {auc_score:.2f})")
                        plt.savefig(results_dir / f"sample_{i}_auc_{auc_score:.2f}.png")
                        plt.close()
                        
                except ValueError:
                    pass
        
        # Summary
        mean_auc = np.mean(aucs)
        mean_ap = np.mean(ap_scores)
        
        logger.info(f"Mean AUC: {mean_auc:.4f}")
        logger.info(f"Mean AP: {mean_ap:.4f}")
        
        # 5. Global ROC/PR Plot
        if all_y_true:
            # Downsample for plotting if too large
            if len(all_y_true) > 100000:
                indices = np.random.choice(len(all_y_true), 100000, replace=False)
                y_true_plot = np.array(all_y_true)[indices]
                y_score_plot = np.array(all_y_scores)[indices]
            else:
                y_true_plot = np.array(all_y_true)
                y_score_plot = np.array(all_y_scores)
                
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true_plot, y_score_plot)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {mean_auc:.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')
            plt.grid()
            plt.savefig(results_dir / "roc_curve.png")
            plt.close()
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true_plot, y_score_plot)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'AP = {mean_ap:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            plt.grid()
            plt.savefig(results_dir / "pr_curve.png")
            plt.close()

        with open(results_dir / "results.txt", "w") as f:
            f.write(f"Mean AUC: {mean_auc:.4f}\n")
            f.write(f"Mean AP: {mean_ap:.4f}\n")
        
if __name__ == "__main__":
    def add_custom_args(parser):
        parser.add_argument("--split", type=str, default="TEST", help="Dataset split (TRAIN, VAL, TEST)")
        # Note: --samples arguments default to -1, handled in the function
        parser.add_argument("--samples", type=int, default=-1, help="Number of samples to evaluate (-1 for all)")
        parser.add_argument("--out", type=str, default=None, help="Output directory")

    scripting.logged_main(
        description="Evaluate Anomaly Detection",
        main_fn=evaluate_anomalies,
        add_args_fn=add_custom_args,
        run_topic="anomaly_detection"
    )

  
