import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import numpy as np
import time
from tqdm import tqdm
import sys

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        
        # Heavy metrics (FID, KID) - Initialize only if needed or keep them but use conditionally
        try:
            self.fid_metric = FrechetInceptionDistance(feature=64).to(self.device)
        except (ModuleNotFoundError, ImportError):
            print("WARNING: torch-fidelity not installed or error. FID metric will be disabled.")
            self.fid_metric = None

        try:
            self.kid_metric = KernelInceptionDistance(subset_size=50).to(self.device)
        except (ModuleNotFoundError, ImportError):
            print("WARNING: torch-fidelity not installed or error. KID metric will be disabled.")
            self.kid_metric = None

    def evaluate_model(self, model, data_loader, heavy_metrics=False):
        # Set model to evaluation mode
        for m in model.models.values():
            m.eval()

        # Reset metrics
        self.ssim_metric.reset()
        self.psnr_metric.reset()
        self.lpips_metric.reset()
        
        if heavy_metrics:
            if self.fid_metric: self.fid_metric.reset()
            if self.kid_metric: self.kid_metric.reset()
            
        inference_times = []
        total_metrics = {}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating model (Heavy={heavy_metrics})", disable=not sys.stdout.isatty()):
                real_A, real_B = batch
                real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                if real_A.ndim == 3: real_A = real_A.unsqueeze(1)
                if real_B.ndim == 3: real_B = real_B.unsqueeze(1)

                start_time = time.time()
                if model.name == "cyclegan":
                    fake_B = model.G_AB(real_A)
                elif model.name == "pix2pix":
                    fake_B = model.generator(real_A)
                else:
                    raise ValueError(f"Unknown model name: {model.name}")
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # --- Standard Metrics (SSIM, PSNR, LPIPS) ---
                # Normalize to [0, 1] if needed
                if fake_B.min() < 0: # Assume [-1, 1] range
                    fake_B_norm = (fake_B + 1) / 2
                    real_B_norm = (real_B + 1) / 2
                else:
                    fake_B_norm = fake_B
                    real_B_norm = real_B

                # Clamp to [0, 1]
                fake_B_norm = torch.clamp(fake_B_norm, 0, 1)
                real_B_norm = torch.clamp(real_B_norm, 0, 1)

                # For LPIPS/FID/KID we need 3 channels
                pred_rgb = fake_B_norm.repeat(1, 3, 1, 1)
                y_rgb = real_B_norm.repeat(1, 3, 1, 1)

                self.ssim_metric.update(fake_B_norm, real_B_norm)
                self.psnr_metric.update(fake_B_norm, real_B_norm)
                self.lpips_metric.update(pred_rgb, y_rgb)
                
                # --- Heavy Metrics (FID, KID) ---
                if heavy_metrics:
                    # Convert to uint8 [0, 255] for standard FID/KID usage
                    real_uint8 = (y_rgb * 255).to(torch.uint8)
                    fake_uint8 = (pred_rgb * 255).to(torch.uint8)
                    
                    if self.fid_metric:
                        self.fid_metric.update(real_uint8, real=True)
                        self.fid_metric.update(fake_uint8, real=False)
                    
                    if self.kid_metric:
                        self.kid_metric.update(real_uint8, real=True)
                        self.kid_metric.update(fake_uint8, real=False)
                
                # --- Validation Losses ---
                # Calculate losses similar to training_step but without backprop
                batch_metrics = {}
                if model.name == "cyclegan":
                    # CycleGAN validation losses
                    fake_A = model.G_BA(real_B)
                    rec_A = model.G_BA(fake_B)
                    rec_B = model.G_AB(fake_A)
                    
                    # Adversarial (using MSE per config, here reusing model.adv_loss)
                    # Note: We need discriminators for adv loss
                    D_fake_B = model.D_B(fake_B)
                    D_fake_A = model.D_A(fake_A)
                    # Generator wants D to classify fakes as real (ones)
                    G_AB_loss = model.adv_loss(D_fake_B, torch.ones_like(D_fake_B))
                    G_BA_loss = model.adv_loss(D_fake_A, torch.ones_like(D_fake_A))
                    
                    # Cycle
                    cycle_loss = model.cycle_loss_fn(rec_A, real_A) + model.cycle_loss_fn(rec_B, real_B)
                    
                    # Identity
                    idt_loss = model.identity_loss_fn(model.G_BA(real_A), real_A) + \
                               model.identity_loss_fn(model.G_AB(real_B), real_B)
                    
                    lambda_cycle = model.config['LAMBDA_CYCLE']
                    lambda_id = model.config.get('LAMBDA_IDENTITY', 0.0)

                    G_loss = (G_AB_loss + G_BA_loss + lambda_cycle * cycle_loss + lambda_id * idt_loss)
                    
                    batch_metrics = {
                        'val_G_loss': G_loss.item(),
                        'val_cycle_loss': cycle_loss.item(),
                    }

                elif model.name == "pix2pix":
                    # Pix2Pix validation losses
                    # D_fake required for G loss
                    D_fake = model.discriminator(torch.cat([real_A, fake_B], dim=1))
                    G_fake_loss = model.BCE(D_fake, torch.ones_like(D_fake))
                    L1_loss = model.L1(fake_B, real_B) * model.config['L1_LAMBDA']
                    G_loss = G_fake_loss + L1_loss
                    
                    batch_metrics = {
                        'val_G_loss': G_loss.item(),
                        'val_L1_loss': L1_loss.item(),
                    }

                # Accumulate batch metrics
                for k, v in batch_metrics.items():
                    if k not in total_metrics: total_metrics[k] = []
                    total_metrics[k].append(v)

        # Resent model to training mode
        for m in model.models.values():
            m.train()

        # Compute Final Metrics
        metrics = {
            'ssim': self.ssim_metric.compute().item(),
            'psnr': self.psnr_metric.compute().item(),
            'lpips': self.lpips_metric.compute().item(),
            'inference_time': float(np.mean(inference_times)) if inference_times else 0.0,
        }

        if heavy_metrics:
            if self.fid_metric:
                metrics['fid'] = self.fid_metric.compute().item()
            if self.kid_metric:
                # KID returns (mean, std), we usually care about mean
                kid_mean, kid_std = self.kid_metric.compute()
                metrics['kid_mean'] = kid_mean.item()
                metrics['kid_std'] = kid_std.item()
        
        # Average loss metrics
        for k, v in total_metrics.items():
            metrics[k] = np.mean(v)
            
        return metrics
