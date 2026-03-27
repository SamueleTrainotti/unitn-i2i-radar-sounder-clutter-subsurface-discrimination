# Source Code

This directory contains the main source code for the project, organized into several Python packages.

## Table of Contents

- [Packages Overview](#packages-overview)
  - [`core`](#core)
  - [`dataset`](#dataset)
  - [`models`](#models)
- [How to Extend](#how-to-extend)
- [Navigation](#navigation)

## Packages Overview

### `core`

The `core` package contains the central components of the project, including:

- **Configuration Handling:** Loading and parsing of configuration files.
- **Logging:** Setup and management of logging for the application.
- **Anomaly Injection:** A reusable `RealisticAnomalyInjector` for synthesizing physical anomalies in radar data.

[Go to `core` README](core/README.md)

### `dataset`

The `dataset` package is responsible for all aspects of data handling:

- **Data Loading:** Loading datasets from disk.
- **Preprocessing:** Applying transformations and augmentations to the data.
- **Caching:** Caching processed datasets for faster access.

[Go to `dataset` README](dataset/README.md)

### `models`

The `models` package implements the GAN models and their components:

- **Base Model:** An abstract base class for GAN models.
- **Pix2Pix:** The implementation of the Pix2Pix model.
- **CycleGAN:** The implementation of the CycleGAN model.
- **Architectures:** The generator and discriminator architectures (e.g., U-Net, PatchGAN).

[Go to `models` README](models/README.md)

## How to Extend

To add new functionality, you can create new modules or packages within the `src` directory. For example, to add a new model, you could create a new file in the `models` package that inherits from the `BaseModel` class.

Because the project is installed in editable mode, any new modules or packages will be immediately available for use in the scripts and notebooks.

## Navigation

- [Back to root README](../README.md)
