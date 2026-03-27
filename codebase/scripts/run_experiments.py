import glob
from collections import defaultdict
import os
import yaml
import shutil
import pathlib
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scripting
# Utility to recursively update a dictionary
def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# Core project imports
from core.config import get_config, set_config, load_config
from core.pipelines import run_anomaly_detection_pipeline

from core.config import get_config, set_config
from core.logger import get_logger
from core.pipelines import run_anomaly_detection_pipeline
from core.run_manager import setup_run_environment


def main():
    """
    Runs a series of anomaly detection experiments on one or more samples
    and generates a comparison plot for each sample.
    """
    logger = get_logger()

    # --- 1. Load experiment configurations from the global config ---
    config = get_config()
    experiments_data = config
    
    runner_config = experiments_data.get('runner_config', {})
    experiments = experiments_data.get('experiments', [])

    if not experiments:
        logger.warning("No experiments found in the configuration file.")
        return

    # Handle single or multiple sample indices
    sample_indices = runner_config.get('sample_indices_to_compare', runner_config.get('sample_index_to_compare', None))
    
    if isinstance(sample_indices, int):
        sample_indices = [sample_indices]
    elif isinstance(sample_indices, list) and len(sample_indices) == 0:
        sample_indices = None
    
    base_config_path = runner_config.get('base_config_path', 'scripts/config_files/detection_config.yaml')

    # --- 2. Setup main output directory (now configurable) ---
    output_base_dir = runner_config.get('output_base_dir', 'comparison_results')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_output_dir = os.path.join(output_base_dir, f"{timestamp}_experiment_run")
    os.makedirs(main_output_dir, exist_ok=True)
    logger.info(f"Starting experiment run. Results will be in: {main_output_dir}")
    if sample_indices:
        logger.info(f"Comparing samples at indices: {sample_indices}")
    else:
        logger.info("Comparing ALL samples in the dataset.")

    # --- 3. Load base detection config (once) ---
    with open(base_config_path, 'r') as f:
        base_detection_config = yaml.safe_load(f)
    
    # Initialize dictionaries to hold paths for both plot types
    extended_plot_paths_by_sample = defaultdict(list)
    diff_plot_paths_by_sample = defaultdict(list)

    # --- 4. Loop through experiments ---
    for i, exp in enumerate(experiments):
        exp_name = exp["name"]
        exp_params = exp.get("params", {})
        
        logger.info(f"--- Running Experiment {i+1}/{len(experiments)}: {exp_name} ---")

        # --- 4a. Create and set experiment-specific config ---
        current_detection_config = deepcopy(base_detection_config)
        current_detection_config = update_dict(current_detection_config, exp_params)

        model_path = current_detection_config.get('MODEL_PATH')
        if not model_path:
            raise ValueError(f"MODEL_PATH not found for experiment '{exp_name}'.")
            
        training_config_path = pathlib.Path(model_path).parent / "config.yaml"
        if not training_config_path.exists():
            raise FileNotFoundError(f"Training config not found for experiment '{exp_name}' at {training_config_path}")

        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
            
        exp_config = update_dict(training_config.copy(), current_detection_config)
        exp_config = update_dict(exp_config, experiments_data)

        exp_output_dir = os.path.join(main_output_dir, exp_name)
        
        # Manually set up the run environment
        run_config = deepcopy(exp_config)
        run_config['OUTPUT_DATA']['BASE_DIR'] = exp_output_dir
        run_config['RUN_DIR'] = exp_output_dir
        
        default_dirs = { "LOGS": "logs", "TEMP": "temp", "EVALUATION": "evaluation" }
        output_data_config = run_config.get("OUTPUT_DATA", {})
        for key, default_name in default_dirs.items():
            dir_name = output_data_config.get(key, default_name)
            abs_path = os.path.join(exp_output_dir, dir_name)
            os.makedirs(abs_path, exist_ok=True)
            output_data_config[key] = abs_path
        run_config["OUTPUT_DATA"] = output_data_config
            
        if 'PLOTTING' not in run_config:
            run_config['PLOTTING'] = {}
        run_config['PLOTTING']['USE_MATPLOTLIB'] = True
        run_config['PLOTTING']['GENERATE_EXTENDED_PLOT'] = True
        run_config['PLOTTING']['SHOW_TITLES'] = True
        
        os.makedirs(exp_output_dir, exist_ok=True)
        with open(os.path.join(exp_output_dir, "config.yaml"), 'w') as f:
            yaml.dump(run_config, f, default_flow_style=False)

        set_config(run_config)

        # --- 4b. Run the detection pipeline ---
        eval_dir = run_config["OUTPUT_DATA"]["EVALUATION"]
        
        # Helper to collect plots for a specific sample index
        def collect_plots_for_sample(s_idx):
            # Store extended comparison plot
            extended_plot_filename = f"{s_idx:04d}_extended_comparison.png"
            extended_plot_path = os.path.join(eval_dir, extended_plot_filename)
            if os.path.exists(extended_plot_path):
                extended_plot_paths_by_sample[s_idx].append({"name": exp_name, "path": extended_plot_path})
            else:
                logger.warning(f"    Extended plot not found for {exp_name} (Sample {s_idx})")

            # Store matplotlib difference plot
            diff_plot_filename = f"{s_idx:04d}_matplotlib_comparison.png"
            diff_plot_path = os.path.join(eval_dir, diff_plot_filename)
            if os.path.exists(diff_plot_path):
                diff_plot_paths_by_sample[s_idx].append({"name": exp_name, "path": diff_plot_path})
            else:
                logger.warning(f"    Difference plot not found for {exp_name} (Sample {s_idx})")

        if sample_indices is not None:
            # Run only for specific samples
            for sample_index in sample_indices:
                try:
                    logger.info(f"  - Processing sample {sample_index}...")
                    run_anomaly_detection_pipeline(limit_to_sample=sample_index)
                    collect_plots_for_sample(sample_index)
                except Exception as e:
                    logger.error(f"    ERROR running experiment {exp_name} on sample {sample_index}: {e}", exc_info=True)
                    continue
        else:
            # Run for ALL samples
            try:
                logger.info(f"  - Processing ALL samples...")
                run_anomaly_detection_pipeline(limit_to_sample=None)
                
                # Discover generated plots
                found_plots = glob.glob(os.path.join(eval_dir, "*_extended_comparison.png"))
                for plot_path in found_plots:
                    try:
                        filename = os.path.basename(plot_path)
                        # Expecting format: 0000_extended_comparison.png
                        idx_str = filename.split('_')[0]
                        s_idx = int(idx_str)
                        collect_plots_for_sample(s_idx)
                    except ValueError:
                        continue
            except Exception as e:
                logger.error(f"    ERROR running experiment {exp_name} on dataset: {e}", exc_info=True)
                continue

    # --- 5. Generate Master Comparison Plot(s) ---
    def generate_master_plot(plot_paths_dict, plot_type_name, output_filename_prefix):
        for sample_index, plot_paths in plot_paths_dict.items():
            if not plot_paths:
                logger.warning(f"No {plot_type_name} plots were generated for sample {sample_index}. Cannot create a master comparison.")
                continue

            logger.info(f"--- Generating Master {plot_type_name} Comparison Plot for Sample {sample_index} ---")
            num_plots = len(plot_paths)
            fig, axs = plt.subplots(num_plots, 1, figsize=(20, 15 * num_plots))
            if num_plots == 1:
                axs = [axs]

            for i, plot_info in enumerate(plot_paths):
                try:
                    img = mpimg.imread(plot_info["path"])
                    axs[i].imshow(img)
                    axs[i].set_title(f"Experiment: {plot_info['name']}", fontsize=16, pad=20)
                    axs[i].axis('off')
                except FileNotFoundError:
                    logger.error(f"Could not find plot file: {plot_info['path']} for master comparison.")
                    axs[i].text(0.5, 0.5, f"Image not found:\n{plot_info['path']}", ha='center', va='center')
                    axs[i].set_title(f"Experiment: {plot_info['name']} (Image Missing)", fontsize=16, pad=20)
                    axs[i].axis('off')

            fig.suptitle(f"Experiment {plot_type_name} Comparison for Sample {sample_index}", fontsize=24, y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            
            final_plot_path = os.path.join(main_output_dir, f"{output_filename_prefix}_{sample_index:04d}.png")
            plt.savefig(final_plot_path)
            plt.close(fig)
            
            logger.info(f"Master {plot_type_name} comparison plot for sample {sample_index} saved to: {final_plot_path}")

    # Generate both types of master plots
    generate_master_plot(extended_plot_paths_by_sample, "Extended", "master_comparison")
    generate_master_plot(diff_plot_paths_by_sample, "Difference", "master_diff_comparison")
    
    logger.info("Experiment run finished.")


if __name__ == "__main__":
    scripting.logged_main(
        "Run a series of anomaly detection experiments",
        main,
        skip_run_setup=True # This script handles its own environment setup
    )