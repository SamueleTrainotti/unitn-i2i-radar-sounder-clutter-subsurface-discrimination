# Dataset Package

This package handles all aspects of data loading, processing, and caching. It is designed to be flexible and efficient, allowing for easy integration of new datasets and preprocessing steps.

## Table of Contents

- [Modules Overview](#modules-overview)
  - [`mapdataset.py`](#mapdatasetpy)
  - [`processing.py`](#processingpy)
  - [`caching.py`](#cachingpy)
  - [`validation.py`](#validationpy)
- [How to Add a New Dataset](#how-to-add-a-new-dataset)
- [Navigation](#navigation)

## Modules Overview

### `mapdataset.py`

This module defines the main `MapDataset` class, which is a custom dataset implementation that maps input images to target images.

**Key Features:**

-   **Two-Pass Normalization:** It performs a two-pass normalization process to ensure global data consistency across the entire dataset.
-   **Caching:** It includes a powerful caching mechanism that saves fully processed datasets, dramatically speeding up subsequent runs by avoiding redundant preprocessing.

### `processing.py`

This module contains all the data preprocessing and augmentation functions. These functions are applied to the images before they are fed to the model.

### `caching.py`

To speed up training, this module provides functionality to cache the processed datasets. This avoids the need to re-process the data every time the training script is run.

### `validation.py`

This module contains functions for validating the integrity and correctness of the datasets.

## How to Add a New Dataset

To add a new dataset, you will typically need to:

1.  Create a new data loading function or class that can read your dataset format.
2.  Define the necessary preprocessing steps in `processing.py`.
3.  Update the configuration file to point to your new dataset and specify the preprocessing pipeline.

## Navigation

- [Back to `src` README](../README.md)
- [Back to root README](../../README.md)
