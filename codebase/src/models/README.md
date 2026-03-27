# Models Package

This package contains the implementations of the GAN models and their constituent architectures.

## Table of Contents

- [Models Overview](#models-overview)
  - [`base_model.py`](#base_modelpy)
  - [`pix2pix.py`](#pix2pixpy)
  - [`cyclegan.py`](#cycleganpy)
- [Architectures](#architectures)
  - [`unet_generator.py`](#unet_generatorpy)
  - [`resnet_generator.py`](#resnet_generatorpy)
  - [`patchgan.py`](#patchganpy)
- [How to Add a New Model](#how-to-add-a-new-model)
- [Navigation](#navigation)

## Models Overview

### `base_model.py`

This module defines the `BaseModel` class, which is an abstract base class for all GAN models in this project. It provides a common interface for training, testing, and saving models.

### `pix2pix.py`

This module contains the implementation of the Pix2Pix model, a conditional GAN for image-to-image translation.

### `cyclegan.py`

This module contains the implementation of the CycleGAN model, which can be used for unpaired image-to-image translation.

## Architectures

The `architectures` subpackage contains the building blocks for the GAN models, including the generator and discriminator networks.

### `unet_generator.py`

This module implements the U-Net generator, which is commonly used in Pix2Pix models.

### `resnet_generator.py`

This module implements a ResNet-based generator, which is often used in CycleGAN models.

### `patchgan.py`

This module implements the PatchGAN discriminator, a network that classifies patches of an image as real or fake.

## How to Add a New Model

To add a new model, you can follow these steps:

1.  Create a new Python file in the `models` directory (e.g., `new_model.py`).
2.  In this file, create a class that inherits from `BaseModel`.
3.  Implement the necessary methods, such as `__init__`, `train_step`, and `test_step`.
4.  Define the generator and discriminator architectures in the `architectures` subpackage if needed.
5.  Update the configuration file to use your new model.

## Navigation

- [Back to `src` README](../README.md)
- [Back to root README](../../README.md)
