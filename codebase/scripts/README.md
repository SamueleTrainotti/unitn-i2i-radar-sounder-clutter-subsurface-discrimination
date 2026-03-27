# Scripts

This directory contains all the runnable Python scripts for the project. These scripts are used for training, testing, hyperparameter tuning, and model comparison.

## Table of Contents

- [Scripts Overview](#scripts-overview)
  - [`train.py`](#trainpy)
  - [`test.py`](#testpy)
  - [`tune_hyperparameters.py`](#tune_hyperparameterspy)
  - [`compare_models.py`](#compare_modelspy)
  - [`benchmark_models.py`](#benchmark_modelspy)
  - [`detect_anomalies.py`](#detect_anomaliespy)
  - [`evaluate_anomalies.py`](#evaluate_anomaliespy)
  - [`generate_anomaly_thesis_plots.py`](#generate_anomaly_thesis_plotspy)
  - [`run_experiments.py`](#run_experimentspy)
- [Configuration Files](#configuration-files)
- [How to Use](#how-to-use)
- [Navigation](#navigation)

## Scripts Overview

### `train.py`

This script is used to train the GAN models. It takes a configuration file as input, which specifies the dataset, model, and training parameters.

### `test.py`

This script is used to test a trained model. It loads a pre-trained model checkpoint and evaluates its performance on a test dataset.

### `tune_hyperparameters.py`

This script performs hyperparameter tuning to find the optimal set of parameters for a model. It uses a search strategy defined in the configuration file.

### `compare_models.py`

This script provides a comprehensive toolkit for evaluating and comparing the performance of different trained models. It takes one or more `run_ids` as input, loads the corresponding models, and runs them on the test dataset. It then generates a detailed report with quantitative metrics and qualitative visual comparisons.

This is essential for:
-   Benchmarking different model architectures (e.g., Pix2Pix vs. CycleGAN).
-   Analyzing the impact of different hyperparameter settings.
-   Visually inspecting the model's output quality on unseen data.

**Key Features:**

-   **Quantitative Evaluation:** Calculates key image quality metrics (PSNR, SSIM, LPIPS) for each model.
-   **Visual Comparison:** Generates images that place the input, ground truth, and each model's output side-by-side for easy comparison.
-   **Metric Visualization:** Creates bar charts and a summary table to visualize and compare the performance metrics across models.
-   **Detailed Reporting:** Saves all results, visualizations, and configuration details in a dedicated output directory.

**Generated Output:**

When you run the script, it creates a `comparison_results` directory (or a custom one if specified) with the following structure:

```
comparison_results/
├── image_comparisons/
│   ├── comparison_sample_0.png
│   ├── comparison_sample_1.png
│   └── ...
├── comparison_metrics.json
├── comparison_summary_table.png
├── LPIPS_comparison.png
├── PSNR_comparison.png
└── SSIM_comparison.png
```

-   `image_comparisons/`: Contains the side-by-side visual comparisons for a few sample images from the test set.
-   `comparison_metrics.json`: A JSON file containing the detailed evaluation metrics for each model.
-   `comparison_summary_table.png`: A table summarizing the key metrics for all compared models.
-   `*_comparison.png`: Bar plots for each metric, comparing the performance of the models.

### `benchmark_models.py`

This script provides a standardized benchmark for evaluating trained models on a specific dataset split (Test or Val). Unlike `compare_models.py`, which is designed for side-by-side comparison, this script allows for independent, rigorous evaluation of single or multiple models, generating scientific plots and CSV reports.

**Key Features:**
-   **Distribution Plots**: Generates violin/box plots for SSIM, PSNR, and LPIPS distributions, allowing for statistical analysis of model performance.
-   **Standardized Grid**: Saves visualization samples in a clean "Input | Target | Generated" grid format.
-   **CSV Reporting**: Exports per-sample metrics to CSV for external analysis.

```bash
python scripts/benchmark_models.py --run-ids <run_id> --split TEST
```

### `evaluate_anomalies.py`

This script quantitatively evaluates the anomaly detection capabilities of the model using **Synthetic Anomaly Injection**. It injects realistic "dipping layers" into the test set and measures the model's ability to highlight them as anomalies.

**Features:**
-   **Synthetic Injection**: Uses the core `RealisticAnomalyInjector` to insert physically plausible anomalies into real radargrams.
-   **ROC/PR Analysis**: Computes Area Under the ROC Curve (AUC) and Average Precision (AP) scores.
-   **Curve Plotting**: Generates and saves ROC and Precision-Recall curves.

```bash
python scripts/evaluate_anomalies.py --run-ids <run_id> --samples 100
```

### `generate_anomaly_thesis_plots.py`

This script is specifically designed to generate high-resolution, annotated visualizations for thesis documentation or external presentations. It focuses on isolating the steps of the anomaly detection pipeline to clearly illustrate how the model operates.

**Features:**
-   **Metric Comparison**: Generates a side-by-side heatmap comparison of the Absolute Difference (L1), Squared Difference (L2), and Structural Dissimilarity (1 - SSIM).
-   **Sliding Window Effect**: Visualizes the noise-reduction effect of applying Average Pooling with increasing kernel sizes (3x3 up to 31x31) over the raw spatial difference map.
-   **Threshold Application**: Demonstrates the conversion from a continuous difference map to a binary anomaly mask using various threshold values (e.g., 0.05 to 0.4).

```bash
python scripts/generate_anomaly_thesis_plots.py --conf scripts/config_files/config.yaml --run_ids <run_id> --split TEST --samples 3 --export-npz
```

### `detect_anomalies.py`

This script uses a trained generator to detect anomalies in radargrams by analyzing the difference between a model's reconstructed output and the ground truth.

#### How it Works

The script loads a trained model and a dataset. For each sample, it performs the following steps:

1.  **Reconstruction:** A trained generator takes a simulated radargram (without anomalies) and attempts to reconstruct the corresponding real-world radargram.
2.  **Difference Calculation:** The reconstructed image is compared against the ground truth real image. The difference can be quantified using a selection of metrics, which are configurable in `detection_config.yaml`.
    *   **Available Metrics:** `L1`, `L2`, `SSIM`, `LPIPS`.
3.  **Calculation Method:** The method for calculating these metrics is also configurable:
    *   `pixel-level`: Computes the metric for each pixel.
    *   `whole-image`: Computes a single metric for the entire image.
    *   `patch-based`: Divides the image into patches and computes the metric for each patch.
    *   `sliding-window`: Computes a moving average over the difference map for smoother results.
4.  **Anomaly Visualization:** The differences are visualized as heatmaps. The script generates detailed plots for each image, including the input, output, difference maps for each selected metric, and binary anomaly masks.
5.  **Summary Report:** A summary plot is generated, showing the average difference map for each metric across all processed images.

This process helps identify areas where the model's reconstruction fails, which often correspond to anomalies or unexpected features in the input data.

#### Configuration

The `detection_config.yaml` file centralizes all settings for the anomaly detection task. Critically, to ensure consistency, this configuration is **merged with the original training configuration** (`config.yaml`) of the model being used. This guarantees that data preprocessing steps (like normalization and patching) are identical to when the model was trained.

Key parameters in `detection_config.yaml` include:

-   `MODEL_PATH`: The absolute path to the directory containing the saved model checkpoints.
-   `INPUT_DATA`: The path to the test data.
-   `OUTPUT_DATA`: The base directory for anomaly detection runs.
-   `THRESHOLD`: The threshold for detecting anomalies.
-   `ANOMALY_DETECTION`: A section to control the detection process:
    -   `METRICS`: A list of metrics to use (e.g., `["L1", "SSIM"]`).
    -   `METHOD`: The calculation method (`pixel-level`, `whole-image`, `patch-based`, or `sliding-window`).
    -   `BACKGROUND_REMOVAL`: A section to control the background removal process:
        -   `METHOD`: The method to remove background noise (`none`, `pca`, or `avg_trace`).
        -   `PCA_COMPONENTS`: The number of principal components to keep for PCA.
-   `PLOTTING`: A section to control plotting, including `USE_MATPLOTLIB`, `SHOW_TITLES`, and `GENERATE_SUMMARY`.

#### Usage

All configuration for the script is handled through the dedicated configuration file.

```bash
python scripts/detect_anomalies.py --config scripts/config_files/detection_config.yaml
```


## Configuration Files

The `config_files` subdirectory contains YAML configuration files for the scripts.

- `config.yaml`: The main configuration file for training and testing.
- `tune_config.yaml`: The configuration file for hyperparameter tuning.
- `detection_config.yaml`: The configuration file for anomaly detection.
- `experiments.yaml`: Defines the set of experiments for `run_experiments.py`.

These files allow you to easily manage and modify the parameters for your experiments.

## How to Use

All scripts are run from the root of the project directory.

### Training a Model

To train a model, use the `train.py` script. The configuration for the training is specified in `scripts/config_files/config.yaml`.

**Starting a New Training Run**

Ensure the `RUN_TO_RESUME` field in `config.yaml` is empty:

```yaml
RUN_TO_RESUME: ""
```

Then, run the following command:

```bash
python scripts/train.py -c scripts/config_files/config.yaml
```

This will create a new, timestamped directory for the run inside the `runs` folder.

**Resuming a Training Run**

1.  Find the ID of the run you want to resume from the `runs` directory (e.g., `2023-10-28_15-30-00_pix2pix`).
2.  Open `scripts/config_files/config.yaml` and set the `RUN_TO_RESUME` field:

```yaml
RUN_TO_RESUME: "2023-10-28_15-30-00_pix2pix"
```

3.  Run the training script:

```bash
python scripts/train.py -c scripts/config_files/config.yaml
```

This will load the latest checkpoint from the specified run and continue training.

### Tuning Hyperparameters

To run hyperparameter tuning, use the `tune_hyperparameters.py` script. This script requires both the main config and a tuning-specific config.

```bash
python scripts/tune_hyperparameters.py -c scripts/config_files/config.yaml -t scripts/config_files/tune_config.yaml
```

This will start an Optuna study to find the best hyperparameters based on the search space defined in `tune_config.yaml`.

### Comparing Models

To compare the performance of models from different training runs, use the `compare_models.py` script with the `--run-ids` argument.

1.  Find the IDs of the runs you want to compare from the `runs` directory.
2.  Run the following command, providing the run IDs after the `--run-ids` flag:

```bash
python scripts/compare_models.py --conf scripts/config_files/config.yaml --run-ids <run_id_1> <run_id_2>
```

For example:

```bash
python scripts/compare_models.py --conf scripts/config_files/config.yaml --run-ids 2023-10-28_15-30-00_pix2pix 2023-10-28_16-00-00_cyclegan
```

The script will generate a detailed comparison report in a new run directory, including metrics and visualizations.

### `run_experiments.py`

This script provides an efficient way to compare the effects of different data processing configurations for anomaly detection. It automates the tedious process of manually changing parameters, running the detection, and comparing outputs across multiple experiments and multiple sample images.

#### How it Works

The script runs a series of "experiments" defined in a single configuration file (e.g., `scripts/config_files/experiments.yaml`). For each experiment, it:
1.  Loads a base configuration specified in the `runner_config` section of the experiments file.
2.  Overrides specific parameters (e.g., enabling TVG, changing the background removal method) as defined for the current experiment.
3.  Runs the anomaly detection pipeline for **one or more sample images**. The indices of the samples to be processed are also defined in the `runner_config`. For each sample, it generates a detailed analysis plot (`<sample_id>_extended_comparison.png`).
4.  After all experiments are complete, it collects the output plots for each sample and stitches them together into separate **`master_comparison_<sample_id>.png`** files.

This provides an at-a-glance view of how different processing steps (TVG, PCA, normalization, etc.) impact the final anomaly detection result, making it ideal for visual analysis and tuning of the preprocessing pipeline across different input images.

#### Configuration

The entire experiment is defined in a single YAML file, typically `scripts/config_files/experiments.yaml`. This file has two main sections:

-   **`runner_config`**: This section contains global settings for the experiment run:
    -   `sample_indices_to_compare`: A list of image indices from the dataset that will be used for comparison across all experiments. Can be a single integer or a list of integers (e.g., `[0, 42, 100]`).
    -   `base_config_path`: The path to the base anomaly detection configuration file (e.g., `scripts/config_files/detection_config.yaml`).
-   **`experiments`**: This is a list where each item represents a single experiment to be run. Each experiment has:
    -   `name`: A unique identifier that will be used for its output folder.
    -   `params`: A dictionary of parameters that will override the values in the `base_config_path`. This is where you define the specific variations for each experiment (e.g., enabling or disabling `Z_SCORE_NORMALIZATION`).

This structure allows you to define a complete comparison matrix in one file.

#### Usage

1.  Set up your desired `runner_config` and define the `experiments` list in a YAML file (e.g., `scripts/config_files/experiments.yaml`). Make sure to specify the sample indices you want to compare.
2.  Run the script from the project root, pointing to your experiments file with the `-c` or `--config` flag:
    ```bash
    python scripts/run_experiments.py -c scripts/config_files/experiments.yaml
    ```
    If no config file is specified, it will default to `scripts/config_files/experiments.yaml`.

The script will create a new timestamped directory in `comparison_results` containing the output for each individual experiment and the final master comparison plots (one for each sample index).

## Navigation

- [Back to root README](../README.md)
