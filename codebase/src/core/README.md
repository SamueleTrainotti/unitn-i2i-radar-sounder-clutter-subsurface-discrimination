# Core Package

This package provides the core functionalities that are shared across the entire project.

## Table of Contents

- [Modules Overview](#modules-overview)
  - [`config.py`](#configpy)
  - [`logger.py`](#loggerpy)
- [How to Use](#how-to-use)
- [Navigation](#navigation)

## Modules Overview

### `config.py`

This module is responsible for loading and managing the project's configuration. It reads the YAML configuration files and provides a simple interface to access the parameters.

### `logger.py`

This module sets up the logging for the application. It provides a pre-configured logger that can be used throughout the project to log messages to the console and/or a file.

### `anomaly_detector.py`

This module contains the `AnomalyDetector` class, which is responsible for detecting anomalies in radargrams. It uses a trained generator to reconstruct an image and then compares the reconstruction to the ground truth to identify differences.

The detector is highly configurable and supports multiple methods and metrics:
-   **Detection Methods**:
    -   `pixel-level`: The raw, pixel-wise difference.
    -   `whole-image`: A single score for the entire image.
    -   `patch-based`: A score computed for each patch in the image.
    -   `sliding-window`: A moving average over the difference map for smoother results.
-   **Difference Metrics**: It can calculate difference maps using standard metrics like `L1` and `L2`, as well as advanced perceptual metrics like `SSIM` and `LPIPS`.

It also generates detailed plots for visualization and analysis.

### `evaluation.py`

This module provides functions for evaluating the performance of the trained models. It includes metrics such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS).

### `run_manager.py`

This module manages the lifecycle of a training or evaluation run. It handles the creation of run directories, saving checkpoints, and logging results with MLflow.

## How to Use

To use the components of this package, you can import them into your modules. For example, to use the logger:

```python
from src.core.logger import logger

logger.info("This is an informational message.")
```

## Navigation

- [Back to `src` README](../README.md)
- [Back to root README](../../README.md)
