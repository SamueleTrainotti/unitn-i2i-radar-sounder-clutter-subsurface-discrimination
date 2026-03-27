# Image-to-Image Translation with GANs

This project implements and explores conditional Generative Adversarial Networks (cGANs), specifically the Pix2Pix model, for image-to-image translation tasks. The codebase is designed to be modular, extensible, and easy to use, allowing for training, testing, and comparison of different models.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
  - [Training](#training)
  - [Testing](#testing)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Model Comparison](#model-comparison)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Project Overview

This repository provides a framework for image-to-image translation using GANs. The primary focus is on the Pix2Pix model, but the architecture is flexible enough to accommodate other models like CycleGAN. The project is structured to facilitate experimentation and research, with clear separation between the core logic, dataset handling, and model implementations.

## Getting Started

### Prerequisites

- Docker
- NVIDIA GPU with CUDA support (for training and inference)

### Installation and Setup

The project is designed to run inside a Docker container to ensure a consistent and isolated development environment.

1.  **Build and start the Docker container:**
    Use the provided script to launch the container:

    ```bash
    ./launch_docker.sh
    ```

    This script will build the Docker image and start a container with the project directory mounted.

2.  **Editable Install:**
    The `entrypoint.sh` script, which runs automatically when the container starts, installs the project in editable mode. This means that any changes made to the source code in the `src` directory will be immediately available without needing to reinstall the package.

## How to Run

All scripts are located in the `scripts` directory and are designed to be run from the root of the project.

### Training

To train a model, run the `train.py` script with a configuration file:

```bash
python scripts/train.py --config scripts/config_files/config.yaml
```

The configuration file (`config.yaml`) allows you to specify dataset paths, model hyperparameters, and training options.

### Testing

To test a trained model, use the `test.py` script:

```bash
python scripts/test.py --config scripts/config_files/config.yaml
```

Make sure to update the configuration file with the path to the pre-trained model checkpoint.

### Hyperparameter Tuning

The `tune_hyperparameters.py` script can be used to find the optimal hyperparameters for a model:

```bash
python scripts/tune_hyperparameters.py --config scripts/config_files/tune_config.yaml
```

### Model Comparison

To compare the performance of different models, use the `compare_models.py` script. This script can evaluate multiple trained models, generate visual comparisons, and provide a detailed report of their performance metrics.

```bash
python scripts/compare_models.py --runs <run_id_1> <run_id_2>
```

For detailed instructions and an explanation of the generated outputs, please refer to the [Scripts README](scripts/README.md#compare_modelspy).

### Experiment Comparison

To efficiently compare the impact of different data processing pipelines on anomaly detection, use the `run_experiments.py` script. This automates running multiple configurations (defined in `scripts/config_files/experiments.yaml`) and generates a single, combined plot for easy visual analysis.

```bash
python scripts/run_experiments.py
```

For detailed instructions, see the [Scripts README](scripts/README.md#run_experimentspy).

### Anomaly Detection

The `detect_anomalies.py` script leverages a trained generator to identify anomalies in new, unseen radargrams. The process is as follows:

1.  **Configuration:** The script uses a dedicated configuration file, `scripts/config_files/detection_config.yaml`. Critically, to ensure consistency, this configuration is merged with the original training configuration (`config.yaml`) of the model being used. This guarantees that data preprocessing steps (like normalization and patching) are identical to when the model was trained.

2.  **Reconstruction:** A trained generator takes a simulated radargram (without anomalies) and attempts to reconstruct the corresponding real-world radargram.

3.  **Difference Calculation:** The reconstructed image is compared against the ground truth (the real image). The difference between them can be quantified using a selection of metrics, which can be configured in the `detection_config.yaml` file. The available metrics are:
    *   **L1 Distance:** The absolute difference between pixel values.
    *   **L2 Distance:** The squared difference between pixel values.
    *   **SSIM (Structural Similarity Index):** A perception-based metric that measures image structural similarity.
    *   **LPIPS (Learned Perceptual Image Patch Similarity):** A deep-learning-based metric that is excellent at capturing human-perceived similarity.

4.  **Calculation Method:** The anomaly score can be calculated using different methods, configurable via the `ANOMALY_DETECTION.METHOD` key:
    *   `pixel-level`: Computes the raw difference map for each pixel.
    *   `whole-image`: Computes a single, mean score for the entire image.
    *   `patch-based`: Divides the image into patches and computes a score for each patch.
    *   `sliding-window`: A new method that computes a moving average over the difference map for smoother, more robust anomaly detection.

5.  **Anomaly Visualization:** The differences are visualized as heatmaps. For each image, the script generates a detailed Matplotlib plot showing the input, ground truth, reconstruction, all computed difference maps, and the final anomaly masks. This is ideal for debugging and analysis.

6.  **Summary Report:** After processing all images, a summary plot (`summary_average_anomalies.png`) is generated. This plot shows the average of the difference maps across the entire test set, providing a high-level view of systematic model behavior.

To run anomaly detection, first configure your desired `METRICS` and `METHOD` in `scripts/config_files/detection_config.yaml`, then run the script:

```bash
python scripts/detect_anomalies.py --config scripts/config_files/detection_config.yaml
```


## Project Structure

The repository is organized as follows:

```
├── Dockerfile           # Defines the Docker environment
├── entrypoint.sh        # Sets up the container environment
├── launch_docker.sh     # Script to build and run the Docker container
├── pyproject.toml       # Project configuration and dependencies
├── README.md            # This file
├── notebooks/           # Jupyter notebooks for exploration and analysis
├── scripts/             # Runnable scripts for training, testing, etc.
└── src/                 # Main source code for the project
```

## Documentation

Detailed documentation for each component of the project can be found in the `README.md` file within each respective directory.

- [Root README](README.md)
- [Scripts README](scripts/README.md)
- [Source Code README](src/README.md)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
