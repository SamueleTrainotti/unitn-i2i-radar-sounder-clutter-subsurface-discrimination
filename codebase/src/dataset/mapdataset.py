"""MapDataset module for handling image-to-image translation datasets."""

import numpy as np # type: ignore
import os
import glob
import sys
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
import psutil # type: ignore
import gc
from tqdm import tqdm # type: ignore
import pdr # type: ignore
from collections import Counter
import logging


from .processing import to_decibels, normalize, get_identifier, extract_patches, handle_irregular_data
from .caching import (get_cache_path, load_from_cache, save_to_cache, two_pass_incremental_build,
                     save_patch_checkpoint, should_free_memory,
                     consolidate_and_free_memory, finalize_dataset)
from .validation import validate_dataset
from utils import save_compare_distributions, save_float_image, save_image, save_pixel_distribution
from core import get_logger, get_config
from scripting import ensure_folder_exists, clear_folder


class MapDataset(Dataset):
    """
    A custom dataset class for loading paired images for image-to-image translation tasks.

    Attributes:
        root_dir (str): Path to the root directory containing the dataset.
        temp_dir (str): Temporary directory used during dataset processing.
        temp_dataset (str): Path to the temporary dataset.
        SAVE_DATASET (bool): Whether to save the processed dataset.
        LOAD_DATASET (bool): Whether to load an existing processed dataset.
    """
    def __init__(self, root_dir, temp_dir="/tmp", temp_dataset="/tmp/dataset", 
                 SAVE_DATASET=True, LOAD_DATASET=False, 
                 incremental_build=True, checkpoint_every=5, patch_size=256, patch_overlap=128,
                 normalization_type='range_zero_to_one',
                 augmentation_config=None,
                 normalization_config=None):
        """
        Initialize the MapDataset.

        Args:
            root_dir (str): Path to the root directory containing the dataset.
            temp_dir (str, optional): Temporary directory used during processing.
            temp_dataset (str, optional): Temporary dataset path.
            SAVE_DATASET (bool, optional): Flag to save dataset after processing.
            LOAD_DATASET (bool, optional): Flag to load existing dataset.
            incremental_build (bool, optional): Whether to build the dataset incrementally.
            checkpoint_every (int, optional): Frequency of checkpoints during incremental build.
            patch_size (int, optional): The size of the patches to extract.
            patch_overlap (int, optional): The overlap between patches.
            normalization_type (str, optional): The normalization type.
            normalization_config (dict, optional): Configuration for normalization, including fixed values.
        Raises:
            - ValueError: If the number of real and simulated images do not match.
            - AssertionError: If the shapes of real and simulated images do not match.
            - AssertionError: If the number of real and simulated images is not equal.
        """        
        self.root_dir = root_dir
        self.temp_dir = temp_dir
        self.temp_dataset = temp_dataset
        self.incremental_build = incremental_build
        self.checkpoint_every = checkpoint_every
        self.logger = get_logger()
        self.logger.info(f"{'=' * 10} Initializing MapDataset {'=' * 10}")
        self.data_dirs = {
            "real": os.path.join(self.root_dir, "real"),
            "sim": os.path.join(self.root_dir, "sim")
        }
        self.data_real = []
        self.data_sim = []
        self.data_min = {"real": +1e20, "sim": +1e20}
        self.data_max = {"real": -1e20, "sim": -1e20}
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.normalization_type = normalization_type
        self.augmentation_config = augmentation_config
        self.fixed_normalization = False
        if normalization_config and normalization_config.get('TYPE') == 'fixed':
            self.fixed_normalization = True
            fixed_values = normalization_config.get('FIXED_VALUES', {})
            self.data_min["real"] = fixed_values.get('REAL_MIN', self.data_min["real"])
            self.data_max["real"] = fixed_values.get('REAL_MAX', self.data_max["real"])
            self.data_min["sim"] = fixed_values.get('SIM_MIN', self.data_min["sim"])
            self.data_max["sim"] = fixed_values.get('SIM_MAX', self.data_max["sim"])
            self.logger.info("Using fixed normalization values.")
            self.logger.info(f"  Real: min={self.data_min['real']:.4f}, max={self.data_max['real']:.4f}")
            self.logger.info(f"  Sim:  min={self.data_min['sim']:.4f}, max={self.data_max['sim']:.4f}")
            
        self.dataset_config = {
            'patch_size': self.patch_size,
            'patch_overlap': self.patch_overlap,
            'normalization_type': self.normalization_type,
        }
        
        ensure_folder_exists(self.temp_dir)
        ensure_folder_exists(self.temp_dataset)
        
        cache_path = get_cache_path(self.root_dir, self.dataset_config)
        
        loaded_from_cache = False
        if LOAD_DATASET:
            if os.path.isfile(cache_path):
                try:
                    if self.fixed_normalization:
                        self.data_real, self.data_sim, _, _, _, self.stats = load_from_cache(cache_path, self.logger)
                        self.logger.info("Successfully loaded dataset from cache, retaining fixed normalization values.")
                    else:
                        self.data_real, self.data_sim, self.data_min, self.data_max, _, self.stats = load_from_cache(cache_path, self.logger)
                        self.logger.info("Successfully loaded dataset from cache.")
                    loaded_from_cache = True
                except Exception as e:
                    self.logger.warning(f"Failed to load dataset from cache: {e}. Rebuilding...")
            else:
                self.logger.warning(f"No existing dataset found at {cache_path}. Rebuilding from scratch.")

        if not loaded_from_cache:
            try:
                stats = None
                if self.incremental_build:
                    stats = two_pass_incremental_build(self, cache_path, SAVE_DATASET)
                else:
                    stats = self._process_and_build_dataset()
                    if SAVE_DATASET:
                        save_to_cache(cache_path, self.data_real, self.data_sim, self.data_min, self.data_max, self.dataset_config, self.logger, stats)
            except Exception as e:
                self.logger.error(f"Error processing dataset: {e}")
                raise RuntimeError("Dataset processing failed and no valid cache available.") from e
        else:
            # If loaded from cache, we might want to log existing stats if available
             if hasattr(self, 'stats') and self.stats:
                self.logger.info(f"Loaded dataset stats: {self.stats}")
            
        validate_dataset(self)

    def _process_files_with_global_normalization(self, paired_files, start_idx, 
                                               checkpoint_path, cache_path, save_final):
        """
        Second pass: normalize and extract patches using global statistics.
        """
        total_files = len(paired_files)
        accumulated_patches_real = []
        accumulated_patches_sim = []
        
        progress_bar = tqdm(
            paired_files[start_idx:], 
            desc="Normalization and extraction",
            initial=start_idx,
            total=total_files,
            file=sys.stdout,
            disable=not sys.stdout.isatty()
        )
        
        for idx, (real_file, sim_file, file_id) in enumerate(progress_bar, start=start_idx):
            try:
                patches_real, patches_sim = self._process_single_pair_normalized(
                    real_file, sim_file, file_id
                )
                
                if patches_real and patches_sim:
                    accumulated_patches_real.extend(patches_real)
                    accumulated_patches_sim.extend(patches_sim)
                
                if (idx + 1) % self.checkpoint_every == 0:
                    save_patch_checkpoint(
                        self, accumulated_patches_real, accumulated_patches_sim,
                        checkpoint_path, idx + 1
                    )
                    
                    if should_free_memory():
                        consolidate_and_free_memory(
                            self, accumulated_patches_real, accumulated_patches_sim
                        )
                
                progress_bar.set_postfix({
                    'patches': (sum(t.shape[0] for t in self.data_real) if isinstance(self.data_real, list) else len(self.data_real)) + len(accumulated_patches_real),
                    'mem_mb': psutil.Process().memory_info().rss // 1024 // 1024
                })
                
            except Exception as e:
                progress_bar.write(f"Errore nel processare {file_id}: {e}")
                continue
        
        progress_bar.close()
        
        finalize_dataset(self, accumulated_patches_real, accumulated_patches_sim)
        
        if save_final:
            save_to_cache(cache_path, self.data_real, self.data_sim, self.data_min, self.data_max, self.dataset_config, self.logger)
            self.logger.info(f"Dataset finale salvato in {cache_path}")

    def _process_single_pair_normalized(self, real_file, sim_file, file_id):
        """
        Processa una singola coppia usando le statistiche globali per normalizzazione.
        """
        try:
            real_image = self.load_image(real_file, is_real=True)
            sim_image = self.load_image(sim_file, is_real=False)
            
            if self.logger.isEnabledFor(logging.NOTSET):
                save_pixel_distribution(
                    real_image,
                    filepath=f"{self.temp_dataset}/pre_DB_real_{file_id}.png",
                    title=f"Real Image Distribution before DB - {file_id}"
                )
                save_pixel_distribution(
                    sim_image,
                    filepath=f"{self.temp_dataset}/pre_DB_sim_{file_id}.png",
                    title=f"Sim Image Distribution before DB - {file_id}"
                )
            
            real_db = to_decibels(real_image)
            sim_db = to_decibels(sim_image)
            
            real_norm = normalize(real_db, self.data_min["real"], self.data_max["real"], self.normalization_type)
            sim_norm = normalize(sim_db, self.data_min["sim"], self.data_max["sim"], self.normalization_type)
            
            if self.logger.isEnabledFor(logging.NOTSET):
                save_pixel_distribution(
                    real_norm,
                    filepath=f"{self.temp_dataset}/real_after_norm_{file_id}.png",
                    title=f"Normalized Real Image Distribution - {file_id}"
                )
                save_pixel_distribution(
                    sim_norm,
                    filepath=f"{self.temp_dataset}/sim_after_norm_{file_id}.png",
                    title=f"Normalized Simulated Image Distribution - {file_id}"
                )
            
            if real_norm.shape == sim_norm.shape:
                patches_real, patches_sim = extract_patches(real_norm, sim_norm, self.patch_size, self.patch_overlap)
                return patches_real, patches_sim
            else:
                self.logger.warning(f"Shape mismatch per {file_id}: {real_norm.shape} vs {sim_norm.shape}")
                return [], []
                
        except Exception as e:
            self.logger.error(f"Errore nel processare coppia {file_id}: {e}")
            return [], []
        finally:
            locals_to_clean = ['real_image', 'sim_image', 'real_db', 'sim_db', 
                             'real_norm', 'sim_norm']
            for var_name in locals_to_clean:
                if var_name in locals():
                    del locals()[var_name]
            gc.collect()

    def _analyze_files(self):
        """Analyze available files and their shapes."""
        self.logger.info("Preliminary file analysis...")
        
        real_files = sorted(glob.glob(os.path.join(self.data_dirs["real"], "*.xml")))
        sim_files = sorted(glob.glob(os.path.join(self.data_dirs["sim"], "*.xml")))
        
        if not real_files or not sim_files:
            raise RuntimeError("No XML files found in real or sim directories")
        
        def get_shape_from_file(file_path, is_real):
            try:
                # Load image to get shape, then release memory
                image = self.load_image(file_path, is_real=is_real)
                shape = image.shape
                del image
                gc.collect()
                return shape
            except Exception as e:
                self.logger.warning(f"Could not get shape from {file_path}: {e}")
            return None

        self.logger.info("Extracting shapes from real images...")
        real_ids = {get_identifier(f): (f, get_shape_from_file(f, is_real=True)) for f in tqdm(real_files, desc="Analyzing real images", file=sys.stdout, disable=not sys.stdout.isatty())}
        
        self.logger.info("Extracting shapes from simulated images...")
        sim_ids = {get_identifier(f): (f, get_shape_from_file(f, is_real=False)) for f in tqdm(sim_files, desc="Analyzing sim images", file=sys.stdout, disable=not sys.stdout.isatty())}
        
        common_ids = set(real_ids.keys()) & set(sim_ids.keys())
        
        if not common_ids:
            raise RuntimeError("No matching pairs found between real and sim files")
        
        paired_files = []
        for id_ in sorted(common_ids):
            real_file, real_shape = real_ids[id_]
            sim_file, sim_shape = sim_ids[id_]
            if real_shape and sim_shape and real_shape == sim_shape:
                paired_files.append((real_file, sim_file, id_))
            else:
                self.logger.warning(f"Shape mismatch for {id_}: real={real_shape}, sim={sim_shape}. Skipping.")

        self.logger.info(f"Files found: {len(real_files)} real, {len(sim_files)} sim")
        self.logger.info(f"Valid pairs identified: {len(paired_files)}")

        if not paired_files:
            raise RuntimeError("Nessuna coppia di file trovata")
        
        if self.logger.isEnabledFor(logging.DEBUG):
            for i, (real_f, sim_f, file_id) in enumerate(paired_files[:5]):
                self.logger.debug(f"Pair {i}: {file_id}")
                self.logger.debug(f"  Real: {os.path.basename(real_f)}")
                self.logger.debug(f"  Sim:  {os.path.basename(sim_f)}")
        
        return {
            'paired_files': paired_files,
            'total_real': len(real_files),
            'total_sim': len(sim_files),
            'paired_count': len(paired_files)
        }

    def _update_global_stats(self, real_image, sim_image):
        """Update global min/max statistics."""
        real_min, real_max = real_image.min(), real_image.max()
        sim_min, sim_max = sim_image.min(), sim_image.max()
        
        if real_min < self.data_min["real"]:
            self.data_min["real"] = real_min
        if real_max > self.data_max["real"]:
            self.data_max["real"] = real_max
        if sim_min < self.data_min["sim"]:
            self.data_min["sim"] = sim_min
        if sim_max > self.data_max["sim"]:
            self.data_max["sim"] = sim_max

    def _log_memory_usage(self):
        """Log current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.logger.debug(f"Memory usage: {memory_mb:.1f} MB")

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        if not self.data_real:
            return 0
        if isinstance(self.data_real, list):
            return sum(t.shape[0] for t in self.data_real)
        return len(self.data_real)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input image and the target image as tensors.
        """
        if isinstance(self.data_real, list):
            for i, tensor_chunk in enumerate(self.data_real):
                if index < tensor_chunk.shape[0]:
                    real_image = self.data_real[i][index]
                    sim_image = self.data_sim[i][index]
                    return (sim_image, real_image)
                index -= tensor_chunk.shape[0]
            raise IndexError("Index out of range")
        
        real_image = self.data_real[index]
        sim_image = self.data_sim[index]

        if self.augmentation_config and self.augmentation_config.get('ADD_NOISE', {}).get('ENABLED', False):
            noise_mean = self.augmentation_config['ADD_NOISE'].get('MEAN', 0.0)
            noise_std = self.augmentation_config['ADD_NOISE'].get('STD', 0.01)
            noise = torch.randn_like(sim_image) * noise_std + noise_mean
            
            if self.normalization_type == 'range_minus_one_to_one':
                min_val, max_val = -1.0, 1.0
            else: # range_zero_to_one
                min_val, max_val = 0.0, 1.0
                
            sim_image = torch.clamp(sim_image + noise, min_val, max_val)
        
        return (sim_image, real_image)
    
    def load_image(self, path, is_real=True):
        """
        Load an image from the specified path with robust error handling.

        Args:
            path (str): Path to the image file.
            is_real (bool): Flag indicating whether the image is real or simulated.

        Returns:
            numpy.ndarray: Loaded image as a numpy array.

        Raises:
            ValueError: If the loaded image is empty or has invalid dimensions.
        """
        try:
            if is_real:
                image = pdr.read(path)["MRO_SHARAD_US_Radargram"]
            else:
                image = pdr.read(path)["Combined_Clutter_Simulation"]
            
            image = handle_irregular_data(image, path, self.logger)
            
            if image is None:
                raise ValueError(f"Immagine None dopo processamento da {path}")
            
            if hasattr(image, 'size') and image.size == 0:
                raise ValueError(f"Immagine vuota dopo processamento da {path}")
            
            if hasattr(image, 'shape'):
                if len(image.shape) != 2:
                    raise ValueError(f"Shape immagine non valida {image.shape} per {path}")
                if image.shape[0] == 0 or image.shape[1] == 0:
                    raise ValueError(f"Dimensioni immagine zero {image.shape} per {path}")
                if image.shape[0] > 10000 or image.shape[1] > 10000:
                    raise ValueError(f"Dimensioni immagine sospette {image.shape} per {path}")
            
            return image
            
        except Exception as e:
            file_id = get_identifier(path)
            raise ValueError(f"Errore nel caricamento file {file_id}: {str(e)}") from e