import os
from abc import ABC, abstractmethod
import shutil
import sys

from core import get_logger, get_config, Evaluator
import torch # type: ignore
from torch.utils.data import DataLoader, random_split # type: ignore
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import save_some_examples


class Model(ABC):
    def __init__(self, name: str = "model", training: bool = True):
        self.logger = get_logger()
        self.config = get_config()
        self.device = self.config['DEVICE']
        self.evaluator = Evaluator(self.device)

        # Models and optimizers
        self.models = {}
        self.optimizers = {}

        self.name = name

        self._evolution_samples = []
        self._evolution_input_sample = None
        self._evolution_target_sample = None
        self.history = []
        self.step_history = []
        self.val_loader = None
        self.train_loader = None

        self.logger.info(f"Initializing model <{self.name}> on device <{self.device}>")
        with self.logger.indent():
            self.build_models()
            if training:
                self.configure_optimizers()

        if training:
            # Early stopping
            self.early_stopping_enabled = self.config.get("EARLY_STOPPING_ENABLED", False)
            if self.early_stopping_enabled:
                self.early_stopping_metric = self.config.get("EARLY_STOPPING_METRIC", "val_psnr")
                self.early_stopping_patience = self.config.get("EARLY_STOPPING_PATIENCE", 10)
                self.early_stopping_mode = self.config.get("EARLY_STOPPING_MODE", "max")
                self.early_stopping_smoothing_window = self.config.get("EARLY_STOPPING_SMOOTHING_WINDOW", 1)
                
                self.early_stopping_counter = 0
                self.best_metric = -float('inf') if self.early_stopping_mode == "max" else float('inf')
                self.metric_history = {}
                self.metric_norm_stats = {
                    'val_psnr': {'min': float('inf'), 'max': -float('inf')},
                    'val_ssim': {'min': float('inf'), 'max': -float('inf')},
                    'val_lpips': {'min': float('inf'), 'max': -float('inf')},
                }

                self.logger.info(f"Early stopping enabled. Metric: {self.early_stopping_metric}, Patience: {self.early_stopping_patience}, Mode: {self.early_stopping_mode}, Smoothing: {self.early_stopping_smoothing_window}")

                if self.early_stopping_metric == "combined":
                    self.combined_metric_weights = self.config.get("EARLY_STOPPING_COMBINED_METRIC_WEIGHTS", {})
                    self.logger.info(f"Using combined metric with weights: {self.combined_metric_weights}")

            use_amp = self.config.get("USE_AMP", False)
            self.amp_enabled = use_amp and self.config['DEVICE'].startswith("cuda")
            if use_amp and not self.amp_enabled:
                self.logger.warning("USE_AMP is true in config, but no CUDA device is available. Disabling AMP.")
            if self.amp_enabled:
                self.logger.info("Automatic Mixed Precision (AMP) is enabled.")
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

            # Create a single dataset containing all images
            self.logger.info("Loading and splitting dataset for training and validation.")
            full_dataset = self.get_dataset()

            # Define split sizes for an 80/20 split
            total_size = len(full_dataset)
            train_size = int(total_size * 0.8)
            val_size = total_size - train_size

            # Perform the split with a fixed seed for reproducibility
            train_dataset, val_dataset = random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            self.logger.info(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
        
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config["BATCH_SIZE"],
                shuffle=True,
                num_workers=self.config["NUM_WORKERS"],
                pin_memory=True
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False, # Validation loader should not be shuffled
                num_workers=1,
                pin_memory=True
            ) if val_dataset else None

        self.logger.info(f"Model <{self.name}> initialized successfully.")

    @abstractmethod
    def build_models(self):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def training_step(self, data, scaler: torch.cuda.amp.GradScaler) -> dict:
        pass
    
    @property
    @abstractmethod
    def latest_checkpoint(self) -> str:
        pass

    def save(self, epoch, filename="checkpoint.pth.tar"):
        """
        Save the model and optimizer states to a file.
        Args:
            epoch (int): The current epoch number.
            filename (str): The name of the file to save the checkpoint.
        Returns:
            None
        """
        self.logger.info(f"Saving model to {filename}")
        checkpoint = {
            "epoch": epoch,
            "models": {k: m.state_dict() for k, m in self.models.items()},
            "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
            "history": self.history,
            "step_history": self.step_history,
        }
        torch.save(checkpoint, filename)


            
    def load(self, path=None):
        """
        Load the model and optimizer states from a file or a directory.
        Args:
            path (str, optional): Path to a checkpoint file or a directory containing checkpoints.
                                  If None, uses the 'latest_checkpoint' property.
                                  If a directory, finds the most recent checkpoint within it.
        Returns:
            int: The epoch to start training from.
        """
        checkpoint_file = None
        if path is None:
            self.logger.warning("No path provided for loading model. Attempting to load latest checkpoint.")
            checkpoint_file = self.latest_checkpoint
        elif os.path.isdir(path):
            self.logger.info(f"Path provided is a directory: {path}. Searching for the latest checkpoint.")
            try:
                checkpoints_dir = path
                found_checkpoints = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith(('.pth', '.pth.tar'))]
                if found_checkpoints:
                    checkpoint_file = max(found_checkpoints, key=os.path.getmtime)
                    self.logger.info(f"Found latest checkpoint: {checkpoint_file}")
                else:
                    self.logger.error(f"No checkpoint files found in directory: {checkpoints_dir}. Cannot load model.")
                    return 0
            except FileNotFoundError:
                self.logger.error(f"Checkpoints directory not found: {path}. Cannot load model.")
                return 0
        elif os.path.isfile(path):
            self.logger.info(f"Path provided is a file: {path}.")
            checkpoint_file = path
        else:
            self.logger.warning(f"Provided path '{path}' is not a valid file or directory. Attempting fallback logic.")
            # Fallback for when the path is a non-existent file, to search its directory
            checkpoints_dir = os.path.dirname(path)
            try:
                found_checkpoints = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith(('.pth', '.pth.tar'))]
                if found_checkpoints:
                    checkpoint_file = max(found_checkpoints, key=os.path.getmtime)
                    self.logger.warning(f"Falling back to the most recently modified checkpoint: {checkpoint_file}")
                else:
                    self.logger.error(f"No checkpoint files found in fallback directory: {checkpoints_dir}. Cannot load model.")
                    return 0
            except FileNotFoundError:
                self.logger.error(f"Fallback directory not found: {checkpoints_dir}. Cannot load model.")
                return 0

        if not checkpoint_file or not os.path.isfile(checkpoint_file):
            self.logger.error("Failed to find a valid checkpoint file. Cannot load model.")
            return 0

        self.logger.info(f"Loading model from {checkpoint_file}")
        try:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint '{checkpoint_file}': {e}. Starting fresh.", exc_info=True)
            return 0
        
        for name, model in self.models.items():
            if name in checkpoint["models"]:
                try:
                    model.load_state_dict(checkpoint["models"][name], strict=False)
                    self.logger.info(f"Loaded state for model: {name}")
                except RuntimeError as e:
                    self.logger.error(f"Failed to load state dict for {name}: {e}. Weights might be incompatible.")
            else:
                self.logger.warning(f"No weights found in checkpoint for model: {name}")

        for name, opt in self.optimizers.items():
            if name in checkpoint["optimizers"]:
                try:
                    opt.load_state_dict(checkpoint["optimizers"][name])
                    self.logger.info(f"Loaded state for optimizer: {name}")
                except ValueError as e:
                    self.logger.error(f"Failed to load state dict for optimizer {name}: {e}. Optimizer state might be incompatible.")
            else:
                self.logger.warning(f"No state found in checkpoint for optimizer: {name}")

        if "history" in checkpoint:
            self.history = checkpoint["history"]
            self.logger.info("Loaded training history.")
        
        if "step_history" in checkpoint:
            self.step_history = checkpoint["step_history"]
            self.logger.info("Loaded step history.")

        start_epoch = checkpoint.get("epoch", 0)
        if start_epoch > 0:
            self.logger.info(f"Resuming training from epoch {start_epoch + 1}")
            return start_epoch
        else:
            self.logger.info("Starting training from scratch.")
            return 0

    def train(self):
        self.logger.info("Starting training loop")
        
        log_every_n_steps = self.config.get("LOG_EVERY_N_STEPS", 0)
        if log_every_n_steps > 0:
            self.logger.info(f"Logging metrics every {log_every_n_steps} steps.")

        start_epoch = 0
        # Optionally load model
        if self.config.get("LOAD_MODEL", False):
            start_epoch = self.load()

        if self.val_loader:
            # Get a fixed sample for evolution plotting
            with self.logger.indent():
                self.logger.debug("Getting a fixed sample for evolution plot.")
            self._evolution_input_sample, self._evolution_target_sample = next(iter(self.val_loader))
            # Ensure they are on CPU and we only keep one image from the batch
            self._evolution_input_sample = self._evolution_input_sample[0:1].cpu()
            self._evolution_target_sample = self._evolution_target_sample[0:1].cpu()

        global_step = 0
        for epoch in range(start_epoch, self.config["NUM_EPOCHS"]):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['NUM_EPOCHS']} started")
            
            # Training step
            self.get_generator().train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config['NUM_EPOCHS']}", mininterval=10, ncols=80, disable=self.config.get("DISABLE_TQDM", False))
            epoch_metrics = []
            for data in loop:
                metrics = self.training_step(data, self.scaler)
                epoch_metrics.append(metrics)
                if log_every_n_steps > 0 and global_step % log_every_n_steps == 0:
                    self.step_history.append(metrics)
                
                loop.set_postfix(metrics)
                self.logger.debug(f"Batch metrics: {metrics}")
                global_step += 1

            # Log average training metrics for the epoch
            avg_train_metrics = self.average_metrics(epoch_metrics, "train")
            self.logger.info(f"Epoch {epoch + 1} average training metrics: {self.format_metrics(avg_train_metrics)}")

            # Validation step
            val_every = self.config.get("VAL_EVERY", 1)
            fid_every = self.config.get("FID_EVERY", 5)
            
            # Check if we should run any validation
            run_val = (epoch + 1) % val_every == 0
            run_heavy = (epoch + 1) % fid_every == 0
            
            if self.val_loader and (run_val or run_heavy):
                self.logger.info(f"Running validation for epoch {epoch + 1} (Heavy={run_heavy})")
                
                # If run_heavy is true, we want FID/KID. 
                # If run_val is true but run_heavy is false, we just want standard metrics.
                # If both are true, we want everything (heavy_metrics=True in evaluate_model covers standard + heavy)
                
                val_metrics = self.evaluator.evaluate_model(self, self.val_loader, heavy_metrics=run_heavy)
                avg_val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
                self.logger.info(f"Epoch {epoch + 1} average validation metrics: {self.format_metrics(avg_val_metrics)}")
                # Merge training and validation metrics for history
                avg_train_metrics.update(avg_val_metrics)

            self.history.append(avg_train_metrics)

            # Optionally save model at specified intervals
            if self.config.get("SAVE_MODEL", False) and (epoch + 1) % self.config.get("SAVE_EVERY", 5) == 0:
                filename = self.get_checkpoint_filename(epoch + 1)
                self.save(epoch + 1, filename)
                self.logger.info(f"Model saved at epoch {epoch + 1}")

            # Early stopping check
            if self.early_stopping_enabled and self.val_loader:
                current_metric_val = None

                # Calculate the metric to be monitored
                if self.early_stopping_metric == "combined":
                    # Update normalization stats
                    for metric_name in self.metric_norm_stats.keys():
                        value = avg_train_metrics.get(metric_name)
                        if value is not None:
                            self.metric_norm_stats[metric_name]['min'] = min(self.metric_norm_stats[metric_name]['min'], value)
                            self.metric_norm_stats[metric_name]['max'] = max(self.metric_norm_stats[metric_name]['max'], value)
                    
                    # Calculate combined score
                    combined_score = 0
                    total_weight = 0
                    for metric_name, weight in self.combined_metric_weights.items():
                        if weight > 0:
                            stats = self.metric_norm_stats[f"val_{metric_name}"]
                            value = avg_train_metrics.get(f"val_{metric_name}")
                            if value is not None and (stats['max'] - stats['min']) > 1e-6:
                                # Normalize to [0, 1]
                                norm_value = (value - stats['min']) / (stats['max'] - stats['min'])
                                # Invert metric if lower is better (e.g., lpips)
                                if metric_name == 'lpips':
                                    norm_value = 1 - norm_value
                                combined_score += norm_value * weight
                                total_weight += weight
                    
                    if total_weight > 0:
                        current_metric_val = combined_score / total_weight

                else:
                    current_metric_val = avg_train_metrics.get(self.early_stopping_metric)

                if current_metric_val is not None:
                    # Apply smoothing
                    metric_name_for_history = self.early_stopping_metric
                    if self.metric_history.get(metric_name_for_history) is None:
                        self.metric_history[metric_name_for_history] = []
                    self.metric_history[metric_name_for_history].append(current_metric_val)
                    
                    if self.early_stopping_smoothing_window > 1:
                        history = self.metric_history[metric_name_for_history]
                        if len(history) >= self.early_stopping_smoothing_window:
                            smoothed_metric = sum(history[-self.early_stopping_smoothing_window:]) / self.early_stopping_smoothing_window
                        else:
                            smoothed_metric = None # Not enough data to smooth yet
                    else:
                        smoothed_metric = current_metric_val

                    if smoothed_metric is not None:
                        # Compare with best metric
                        improvement = False
                        if self.early_stopping_mode == "max":
                            if smoothed_metric > self.best_metric:
                                improvement = True
                        else: # min mode
                            if smoothed_metric < self.best_metric:
                                improvement = True

                        if improvement:
                            self.best_metric = smoothed_metric
                            self.early_stopping_counter = 0
                            self.logger.info(f"New best metric: {self.best_metric:.4f}. Saving model.")
                            self.save(epoch + 1, self.latest_checkpoint)
                        else:
                            self.early_stopping_counter += 1
                            self.logger.info(f"No improvement. Best: {self.best_metric:.4f}, Current (smoothed): {smoothed_metric:.4f}, Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

                        if self.early_stopping_counter >= self.early_stopping_patience:
                            self.logger.info(f"Early stopping triggered after {self.early_stopping_patience} epochs with no improvement.")
                            break
                
            # Call validation hook if available
            if self.val_loader and (epoch + 1) % self.config.get("SAVE_EVERY", 5) == 0:
                self.logger.info(f"Generating validation examples at epoch {epoch + 1}")
                self.generate_examples(epoch + 1, self.get_generator())
                
            if self.val_loader and (epoch + 1) % self.config.get("EVOLUTION_EVERY", 5) == 0:
                self.store_evolution_examples(epoch + 1)

        self.logger.info("Training completed")

        # Save the final model
        if self.config.get("SAVE_MODEL", False):
            filename = self.get_checkpoint_filename(self.config["NUM_EPOCHS"])
            self.save(self.config["NUM_EPOCHS"], filename)
            self.logger.info(f"Final model saved at epoch {self.config['NUM_EPOCHS']}")

        if self._evolution_samples:
            from utils import plot_evolution
            output_path = f"{self.config['OUTPUT_DATA']['EVALUATION']}/{self.name}_evolution.png"
            self.logger.info(f"Plotting model evolution to {output_path}")
            plot_evolution(self, output_path)
        
        self.plot_metrics()
        self.plot_step_losses()

    def average_metrics(self, epoch_metrics, prefix=""):
        if not epoch_metrics:
            return {}
        
        # Calculate average for each key present in the metrics
        avg_metrics = {}
        # Get all unique keys from all dictionaries in the list
        all_keys = set().union(*(d.keys() for d in epoch_metrics))

        for k in all_keys:
            values = []
            for m in epoch_metrics:
                if k in m:
                    val = m[k]
                    # Ensure value is a number, converting tensor if necessary
                    if isinstance(val, torch.Tensor):
                        values.append(val.item())
                    elif isinstance(val, (int, float)):
                        values.append(val)
            if values:
                avg_metrics[f"{prefix}_{k}"] = sum(values) / len(values)
        return avg_metrics

    def format_metrics(self, metrics):
        return {k: round(v, 4) for k, v in metrics.items()}

    def plot_training_losses(self):
        """
        Plot training losses with thematic splits: Adversarial and Structural.
        Supports both CycleGAN and Pix2Pix.
        """
        if not self.history:
            return

        epochs = range(1, len(self.history) + 1)
        plotting_config = self.config.get('PLOTTING', {})
        
        # --- Plot A: Adversarial Stability (D_A, D_B, G_AB, G_BA, D, G_fake) ---
        default_adv_keys = [
            'train_D_A_loss', 'train_D_B_loss', 
            'train_G_AB_loss', 'train_G_BA_loss',
            'train_D_loss', 'train_G_fake_loss', 'train_G_fake_loss_prop'
        ]
        
        # Override with config if available
        adv_keys = plotting_config.get('ADVERSARIAL_KEYS', default_adv_keys)
        # Filter keys that actually exist in history
        adv_keys = [k for k in adv_keys if any(k in h for h in self.history)]
        
        if adv_keys:
            plt.figure(figsize=(10, 6))
            for key in adv_keys:
                values = [h.get(key) for h in self.history]
                if any(v is not None for v in values):
                    # Remove 'train_' prefix for cleaner legend
                    label = key.replace('train_', '')
                    plt.plot(epochs, values, label=label, linewidth=1.5)
            
            plt.title("Training Losses - Adversarial Stability", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(loc='best', frameon=True, fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            out_path = f"{self.config['OUTPUT_DATA']['EVALUATION']}/{self.name}_train_loss_adversarial.png"
            plt.savefig(out_path, dpi=300)
            self.logger.info(f"Saved adversarial loss plot to {out_path}")
            plt.close()

        # --- Plot B: Structural & Total Consistency (Cycle, Identity, L1, Total G) ---
        default_struct_keys = [
            'train_G_loss', 'train_cycle_loss', 'train_identity_loss',
            'train_L1_loss', 'train_L1_loss_prop'
        ]
        struct_keys = plotting_config.get('STRUCTURAL_KEYS', default_struct_keys)
        struct_keys = [k for k in struct_keys if any(k in h for h in self.history)]
        
        if struct_keys:
            plt.figure(figsize=(10, 6))
            for key in struct_keys:
                values = [h.get(key) for h in self.history]
                if any(v is not None for v in values):
                    label = key.replace('train_', '')
                    plt.plot(epochs, values, label=label, linewidth=1.5)
            
            plt.title("Training Losses - Structural Consistency", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(loc='best', frameon=True, fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            out_path = f"{self.config['OUTPUT_DATA']['EVALUATION']}/{self.name}_train_loss_structural.png"
            plt.savefig(out_path, dpi=300)
            self.logger.info(f"Saved structural loss plot to {out_path}")
            plt.close()

    def plot_validation_losses(self):
        """
        Plot validation losses separately from training.
        """
        if not self.history:
            return

        epochs = range(1, len(self.history) + 1)
        plotting_config = self.config.get('PLOTTING', {})
        
        default_val_keys = ['val_val_G_loss', 'val_val_cycle_loss', 'val_val_L1_loss', 'val_G_loss', 'val_cycle_loss']
        val_keys = plotting_config.get('VAL_KEYS', default_val_keys)
        val_keys = [k for k in val_keys if any(k in h for h in self.history)]
        
        if not val_keys:
            return

        plt.figure(figsize=(10, 6))
        for key in val_keys:
            values = [h.get(key) for h in self.history]
            valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if valid_values:
                # Clean up label: remove 'val_val_' or 'val_'
                label = key.replace('val_val_', 'val_').replace('val_', '', 1) 
                if label.startswith('val_'): label = label[4:]
                
                plt.plot(valid_epochs, valid_values, label=label, marker='o', linestyle='-', linewidth=1.5, markersize=4)

        plt.title("Validation Losses", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(loc='best', frameon=True, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        out_path = f"{self.config['OUTPUT_DATA']['EVALUATION']}/{self.name}_val_loss.png"
        plt.savefig(out_path, dpi=300)
        self.logger.info(f"Saved validation loss plot to {out_path}")
        plt.close()

    def plot_quality_metrics(self):
        """
        Plot SSIM, PSNR, and LPIPS in a vertical stack (3x1).
        """
        if not self.history:
            return
            
        epochs = range(1, len(self.history) + 1)
        plotting_config = self.config.get('PLOTTING', {})
        
        # Define default metrics
        default_metrics_config = [
            {'name': 'PSNR', 'train_key': 'train_psnr', 'val_key': 'val_psnr', 'ylabel': 'PSNR (dB)', 'color': '#1f77b4'}, # Blue
            {'name': 'SSIM', 'train_key': 'train_ssim', 'val_key': 'val_ssim', 'ylabel': 'SSIM', 'color': '#ff7f0e'},      # Orange
            {'name': 'LPIPS', 'train_key': 'train_lpips', 'val_key': 'val_lpips', 'ylabel': 'LPIPS', 'color': '#2ca02c'}   # Green
        ]
        
        metrics_config = plotting_config.get('QUALITY_METRICS', default_metrics_config)

        fig, axs = plt.subplots(len(metrics_config), 1, figsize=(10, 4 * len(metrics_config)), sharex=True)
        if len(metrics_config) == 1:
            axs = [axs] # Ensure iterable if only one subplot
        
        data_found = False
        
        for i, config in enumerate(metrics_config):
            ax = axs[i]
            train_key = config.get('train_key')
            val_key = config.get('val_key')
            
            # Extract data
            train_vals = [h.get(train_key) for h in self.history] if train_key else [None]*len(self.history)
            val_vals = [h.get(val_key) for h in self.history] if val_key else [None]*len(self.history)
            
            # Check availability
            has_train = any(v is not None for v in train_vals)
            has_val = any(v is not None for v in val_vals)
            
            if has_train or has_val:
                data_found = True
            
            color = config.get('color', '#1f77b4')

            # Plot Training
            if has_train:
                valid_epochs = [e for e, v in zip(epochs, train_vals) if v is not None]
                valid_vals = [v for v in train_vals if v is not None]
                ax.plot(valid_epochs, valid_vals, label=f"Train", linestyle='-', color=color, alpha=0.6)
                
            # Plot Validation
            if has_val:
                valid_epochs = [e for e, v in zip(epochs, val_vals) if v is not None]
                valid_vals = [v for v in val_vals if v is not None]
                ax.plot(valid_epochs, valid_vals, label=f"Val", linestyle='--', marker='.', color=color, linewidth=2)
                
            ax.set_ylabel(config.get('ylabel', ''), fontsize=12)
            ax.legend(loc='upper left', frameon=True, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_title(f"{config.get('name', '')} Evolution", fontsize=12, loc='left', pad=10)

        if not data_found:
            plt.close(fig)
            return

        axs[-1].set_xlabel("Epoch", fontsize=12)
        plt.suptitle(f"Training and Validation Quality Metrics - {self.name}", fontsize=16)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        out_path = f"{self.config['OUTPUT_DATA']['EVALUATION']}/{self.name}_quality_metrics_stack.png"
        plt.savefig(out_path, dpi=300)
        self.logger.info(f"Saved quality metric stack to {out_path}")
        plt.close()

    def plot_metrics(self):
        """
        Plot all training and validation metrics using the new thematic structure.
        """
        if not self.history:
            self.logger.warning("No history to plot.")
            return

        self.plot_training_losses()
        self.plot_validation_losses()
        self.plot_quality_metrics()


    def plot_step_losses(self):
        """
        Plot the training losses per step, split into separate plots to avoid scale squashing,
        matching the style of the per-epoch plots.
        """
        if not self.step_history:
            self.logger.warning("No step history to plot.")
            return

        loss_keys = [k for k in self.step_history[0].keys() if 'loss' in k.lower()]
        
        # Define groupings based on scale
        adv_keywords = ['D_A_loss', 'D_B_loss', 'D_loss', 'G_AB_loss', 'G_BA_loss', 'G_fake_loss', 'G_fake_loss_prop']
        # Everything else goes to structural/total (G_loss, L1, cycle, identity)
        
        adv_keys = [k for k in loss_keys if k in adv_keywords]
        struct_keys = [k for k in loss_keys if k not in adv_keys]
        
        if adv_keys:
            plt.figure(figsize=(10, 6))
            for key in adv_keys:
                values = [e.get(key, 0) if not isinstance(e.get(key, 0), torch.Tensor) else e.get(key, 0).item() for e in self.step_history]
                plt.plot(values, label=key, alpha=0.8)
            plt.title("per-Step Losses: Adversarial Components", fontsize=14)
            plt.xlabel("Step", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(loc='best', frameon=True, fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            output_path = f"{self.config['OUTPUT_DATA']['EVALUATION']}/{self.name}_step_losses_adversarial.png"
            plt.savefig(output_path, dpi=300)
            self.logger.info(f"Saved adversarial step loss plot to {output_path}")
            plt.close()
            
        if struct_keys:
            plt.figure(figsize=(10, 6))
            for key in struct_keys:
                values = [e.get(key, 0) if not isinstance(e.get(key, 0), torch.Tensor) else e.get(key, 0).item() for e in self.step_history]
                plt.plot(values, label=key, alpha=0.8)
            plt.title("per-Step Losses: Structural & Total", fontsize=14)
            plt.xlabel("Step", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(loc='best', frameon=True, fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            output_path = f"{self.config['OUTPUT_DATA']['EVALUATION']}/{self.name}_step_losses_structural.png"
            plt.savefig(output_path, dpi=300)
            self.logger.info(f"Saved structural step loss plot to {output_path}")
            plt.close()

    def _create_dataset(self, data_dir):
        from dataset import MapDataset
        return MapDataset(
            root_dir=data_dir,
            temp_dir=self.config["OUTPUT_DATA"]["TEMP"],
            temp_dataset=self.config["OUTPUT_DATA"]["TEMP_DATASET"],
            SAVE_DATASET=self.config["SAVE_DATASET"],
            LOAD_DATASET=self.config["LOAD_DATASET"],
            patch_size=self.config.get("PATCH_SIZE", 256),
            patch_overlap=self.config.get("PATCH_OVERLAP", 128),
            normalization_type=self.config.get("NORMALIZATION_TYPE", "range_zero_to_one"),
            augmentation_config=self.config.get("AUGMENTATION"),
        )

    def get_dataset(self):
        return self._create_dataset(self.config["INPUT_DATA"]["TRAIN"])
        
    @abstractmethod
    def get_generator(self):
        """
        Returns the generator model instance.
        This method should be implemented by subclasses to return the specific generator model.
        """
        raise NotImplementedError("Subclasses must implement get_generator method.")
        
    def get_checkpoint_filename(self, epoch):
        return f"{self.config['OUTPUT_DATA']['CHECKPOINTS']}/{self.name}_epoch_{epoch}.pth"

    def generate_examples(self, epoch, generator):
        if generator is None:
            raise NotImplementedError("Subclasses must pass a Generator instance to generate_examples.")
        folder = self.config['OUTPUT_DATA']['EVALUATION']
        self.logger.info(f"Saving validation examples to {folder} at epoch {epoch}")
        save_some_examples(generator, self.val_loader, epoch, folder, self.device)

    def store_evolution_examples(self, epoch):
        if self._evolution_input_sample is None:
            self.logger.warning("No evolution sample set, skipping.")
            return

        generator = self.get_generator()
        generator.eval()
        with torch.no_grad():
            # The input sample is on CPU, move it to the correct device for prediction
            _input = self._evolution_input_sample.to(self.device)
            # Model expects a channel dimension, so we unsqueeze it
            pred = generator(_input.unsqueeze(1))
            self._evolution_samples.append((epoch, pred.cpu()))
        generator.train()