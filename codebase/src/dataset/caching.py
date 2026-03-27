import os
import torch
import numpy as np
import gc
import psutil
import sys
import hashlib
import json
from tqdm import tqdm
from .processing import to_decibels

def get_cache_path(root_dir, config):
    """Generates a unique cache path based on the dataset configuration."""
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
    return os.path.join(root_dir, f"dataset_{config_hash[:10]}.pt")


def load_from_cache(path, logger):
    """Handles loading the pre-processed .pt file."""
    logger.info(f"Loading existing dataset from <{path}> ...")
    data = torch.load(path)
    loaded_real = data["real"]
    loaded_sim = data["sim"]
    data_min = data.get("min_stats")
    data_max = data.get("max_stats")
    config = data.get("config")

    if data_min is None or data_max is None:
        raise ValueError("Cache file is missing required min/max normalization statistics.")
    
    if torch.is_tensor(loaded_real) and torch.is_tensor(loaded_sim):
        data_real = loaded_real
        data_sim = loaded_sim
    elif isinstance(loaded_real, list) and isinstance(loaded_sim, list):
        data_real = loaded_real
        data_sim = loaded_sim
    elif isinstance(loaded_real, np.ndarray) and isinstance(loaded_sim, np.ndarray):
        data_real = torch.from_numpy(loaded_real)
        data_sim = torch.from_numpy(loaded_sim)
    elif isinstance(loaded_real, list) and isinstance(loaded_sim, list):
        data_real = torch.tensor(loaded_real)
        data_sim = torch.tensor(loaded_sim)
    else:
        raise ValueError("Unsupported data type for loaded dataset.")
    
    if loaded_real is None or loaded_sim is None:
        raise ValueError("Loaded data is None. Cache may be corrupted.")
    if len(loaded_real) == 0 or len(loaded_sim) == 0:
        raise ValueError("Loaded dataset is empty. Cache is likely incomplete.")
    
    
    stats = data.get("stats")
    
    return data_real, data_sim, data_min, data_max, config, stats

def save_to_cache(path, data_real, data_sim, data_min, data_max, config, logger, stats=None):
    """Saves the processed tensors to a .pt file and a .json metadata file."""
    logger.info(f'Saving dataset to {path}')
    temp_path = f"{path}.tmp"
    data_to_save = {
        "real": data_real,
        "sim": data_sim,
        "min_stats": data_min,
        "max_stats": data_max,
        "config": config,
        "stats": stats
    }
    torch.save(data_to_save, temp_path)
    os.replace(temp_path, path)

    # Save metadata as a human-readable JSON file
    metadata_path = os.path.splitext(path)[0] + '.json'
    try:
        metadata = config.copy()
        if stats:
            metadata['dataset_stats'] = stats
            
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f'Metadata saved to {metadata_path}')
    except Exception as e:
        logger.warning(f'Failed to save metadata: {e}')

def two_pass_incremental_build(dataset, cache_path, save_final=True):
    """
    Costruzione incrementale a due passate con checkpoint funzionanti.
    
    Passata 1: Calcola statistiche globali (con possibile recovery)
    Passata 2: Normalizza ed estrae patch con gestione memoria ottimizzata
    """
    dataset.logger.info("Avvio costruzione incrementale a due passate...")
    
    # Step 1: Analisi preliminare dei file
    files_info = dataset._analyze_files()
    total_files = len(files_info['paired_files'])
    
    if total_files == 0:
        raise RuntimeError("Nessuna coppia di file trovata")
    
    dataset.logger.info(f"Trovate {total_files} coppie di file da processare")
    
    # Percorsi dei checkpoint
    stats_checkpoint_path = os.path.join(dataset.temp_dataset, "stats_checkpoint.pt")
    patch_checkpoint_path = os.path.join(dataset.temp_dataset, "patch_checkpoint.pt")
    
    # PASSATA 1: Calcolo statistiche globali con checkpoint
    if not dataset.fixed_normalization:
        stats_completed = False
        stats_start_idx = 0
        
        # Verifica se esistono statistiche da checkpoint
        if os.path.exists(stats_checkpoint_path):
            try:
                stats_completed, stats_start_idx = resume_stats_checkpoint(dataset, stats_checkpoint_path)
                if stats_completed:
                    dataset.logger.info("Statistiche globali già calcolate, caricamento da checkpoint")
                else:
                    dataset.logger.info(f"Ripresa calcolo statistiche dall'immagine {stats_start_idx}/{total_files}")
            except Exception as e:
                dataset.logger.warning(f"Impossibile riprendere checkpoint statistiche: {e}")
                stats_start_idx = 0
        
        if not stats_completed:
            dataset.logger.info("PASSATA 1: Calcolo statistiche globali min/max...")
            calculate_global_statistics_with_checkpoint(
                dataset, files_info['paired_files'], stats_start_idx, stats_checkpoint_path
            )
            
            dataset.logger.info(f"Statistiche globali calcolate:")
            dataset.logger.info(f"  Real: min={dataset.data_min['real']:.4f}, max={dataset.data_max['real']:.4f}")
            dataset.logger.info(f"  Sim:  min={dataset.data_min['sim']:.4f}, max={dataset.data_max['sim']:.4f}")
    else:
        dataset.logger.info("PASSATA 1: Saltata, usando valori di normalizzazione fissi.")
    
    # PASSATA 2: Normalizzazione ed estrazione patch con checkpoint
    dataset.logger.info("PASSATA 2: Normalizzazione ed estrazione patch...")
    
    # Verifica se esiste un checkpoint per la passata 2
    patch_start_idx = 0
    if os.path.exists(patch_checkpoint_path):
        try:
            patch_start_idx = resume_patch_checkpoint(dataset, patch_checkpoint_path)
            dataset.logger.info(f"Ripresa estrazione patch dall'immagine {patch_start_idx}/{total_files}")
        except Exception as e:
            dataset.logger.warning(f"Impossibile riprendere checkpoint patch: {e}")
            patch_start_idx = 0
    
    # Elaborazione incrementale con normalizzazione corretta
    dataset._process_files_with_global_normalization(
        files_info['paired_files'], patch_start_idx, patch_checkpoint_path, cache_path, save_final
    )
    
    # Pulizia checkpoint temporanei
    for checkpoint_file in [stats_checkpoint_path, patch_checkpoint_path]:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            dataset.logger.debug(f"Checkpoint temporaneo rimosso: {os.path.basename(checkpoint_file)}")
            
    # Return collected statistics
    total_patches = sum(t.shape[0] for t in dataset.data_real) if isinstance(dataset.data_real, list) else len(dataset.data_real)
    
    return {
        'total_files_real': files_info['total_real'],
        'total_files_sim': files_info['total_sim'],
        'valid_pairs': files_info['paired_count'],
        'total_patches': total_patches
    }

def calculate_global_statistics_with_checkpoint(dataset, paired_files, start_idx, checkpoint_path):
    """
    Prima passata con checkpoint: calcola le statistiche min/max globali.
    """
    progress_bar = tqdm(
        paired_files[start_idx:], 
        desc="Calcolo statistiche globali",
        initial=start_idx,
        total=len(paired_files),
        file=sys.stdout,
        disable=not sys.stdout.isatty()
    )
    valid_files_count = 0
    skipped_files = []
    
    for idx, (real_file, sim_file, file_id) in enumerate(progress_bar, start=start_idx):
        try:
            # Carica e converti in dB
            real_image = dataset.load_image(real_file, is_real=True)
            sim_image = dataset.load_image(sim_file, is_real=False)
            
            # Verifica dimensioni compatibili
            if real_image.shape != sim_image.shape:
                dataset.logger.warning(f"Shape mismatch per {file_id}: real={real_image.shape}, sim={sim_image.shape}. Skipped.")
                skipped_files.append(file_id)
                continue
            
            real_db = to_decibels(real_image)
            sim_db = to_decibels(sim_image)
            
            # Verifica che i dB siano validi
            if not np.isfinite(real_db).all() or not np.isfinite(sim_db).all():
                dataset.logger.warning(f"Valori non finiti in dB per {file_id}. Skipped.")
                skipped_files.append(file_id)
                continue
            
            # Aggiorna statistiche globali
            dataset._update_global_stats(real_db, sim_db)
            valid_files_count += 1
            
            # Checkpoint periodico per statistiche
            if (idx + 1) % dataset.checkpoint_every == 0:
                save_stats_checkpoint(dataset, checkpoint_path, idx + 1, False)
            
            # Aggiorna progress bar
            progress_bar.set_postfix({
                'valid': valid_files_count,
                'skipped': len(skipped_files),
                'real_range': f"[{dataset.data_min['real']:.1f}, {dataset.data_max['real']:.1f}]",
                'sim_range': f"[{dataset.data_min['sim']:.1f}, {dataset.data_max['sim']:.1f}]"
            })
            
            # Pulizia esplicita
            del real_image, sim_image, real_db, sim_db
            
        except Exception as e:
            dataset.logger.error(f"Errore nel calcolo statistiche per {file_id}: {e}")
            skipped_files.append(file_id)
            continue
            
    progress_bar.close()
    
    # Salva checkpoint finale per statistiche complete
    save_stats_checkpoint(dataset, checkpoint_path, len(paired_files), True)
    
    if skipped_files:
        dataset.logger.warning(f"File saltati ({len(skipped_files)}): {', '.join(skipped_files)}")
    
    if valid_files_count == 0:
        raise RuntimeError("Nessun file valido trovato per il calcolo delle statistiche globali")
    
    dataset.logger.info(f"Statistiche calcolate su {valid_files_count} file validi")
    gc.collect()

def save_stats_checkpoint(dataset, checkpoint_path, processed_count, completed):
    """Salva checkpoint per la fase di calcolo statistiche."""
    try:
        checkpoint_data = {
            'processed_count': processed_count,
            'data_min': dataset.data_min,
            'data_max': dataset.data_max,
            'stats_completed': completed,
            'checkpoint_type': 'statistics'
        }
        
        temp_checkpoint = f"{checkpoint_path}.tmp"
        torch.save(checkpoint_data, temp_checkpoint)
        os.replace(temp_checkpoint, checkpoint_path)
        
        status = "completate" if completed else "in corso"
        dataset.logger.debug(f"Checkpoint statistiche salvato: {processed_count} file, stato={status}")
        
    except Exception as e:
        dataset.logger.warning(f"Errore nel salvataggio checkpoint statistiche: {e}")

def resume_stats_checkpoint(dataset, checkpoint_path):
    """Riprende il calcolo delle statistiche da checkpoint."""
    dataset.logger.info("Caricamento checkpoint statistiche...")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Verifica tipo checkpoint
    if checkpoint.get('checkpoint_type') != 'statistics':
        dataset.logger.warning("Tipo checkpoint non corretto, ignorato")
        return False, 0
    
    # Ripristina statistiche
    dataset.data_min = checkpoint.get('data_min', {"real": +1e20, "sim": +1e20})
    dataset.data_max = checkpoint.get('data_max', {"real": -1e20, "sim": -1e20})
    
    processed_count = checkpoint.get('processed_count', 0)
    stats_completed = checkpoint.get('stats_completed', False)
    
    dataset.logger.info(f"Checkpoint statistiche caricato: {processed_count} file processati")
    if stats_completed:
        dataset.logger.info(f"Statistiche globali: Real[{dataset.data_min['real']:.2f}, {dataset.data_max['real']:.2f}], "
                        f"Sim[{dataset.data_min['sim']:.2f}, {dataset.data_max['sim']:.2f}]")
    
    return stats_completed, processed_count

def save_patch_checkpoint(dataset, patches_real, patches_sim, checkpoint_path, processed_count):
    """Salva checkpoint per la fase di estrazione patch con validazione."""
    try:
        # Validazione e pulizia delle patch prima del salvataggio
        cleaned_patches_real = []
        cleaned_patches_sim = []
        
        # Verifica che le patch siano array numpy regolari
        for i, (patch_real, patch_sim) in enumerate(zip(patches_real, patches_sim)):
            try:
                # Converti in array numpy e verifica forma
                if not isinstance(patch_real, np.ndarray):
                    patch_real = np.array(patch_real)
                if not isinstance(patch_sim, np.ndarray):
                    patch_sim = np.array(patch_sim)
                
                # Verifica che siano bidimensionali e della stessa forma
                if (len(patch_real.shape) == 2 and len(patch_sim.shape) == 2 and 
                    patch_real.shape == patch_sim.shape and
                    patch_real.shape == (dataset.patch_size, dataset.patch_size)):
                    
                    # Verifica che contengano solo numeri finiti
                    if (np.isfinite(patch_real).all() and np.isfinite(patch_sim).all()):
                        cleaned_patches_real.append(patch_real.astype(np.float32))
                        cleaned_patches_sim.append(patch_sim.astype(np.float32))
                    else:
                        dataset.logger.debug(f"Patch {i} for file {processed_count} contains non-finite values, skipped.")
                else:
                    dataset.logger.debug(f"Patch {i} for file {processed_count} has invalid shape: real={patch_real.shape}, sim={patch_sim.shape}")
                    
            except Exception as e:
                dataset.logger.debug(f"Error validating patch {i} for file {processed_count}: {e}")
                continue
        
        if len(cleaned_patches_real) != len(patches_real):
            removed_count = len(patches_real) - len(cleaned_patches_real)
            dataset.logger.warning(f"Removed {removed_count} irregular patches from checkpoint for file {processed_count}")
        
        # Salva solo se ci sono patch valide
        if len(cleaned_patches_real) > 0:
            # Converti in array numpy per serializzazione sicura
            patches_real_array = np.array(cleaned_patches_real, dtype=np.float32)
            patches_sim_array = np.array(cleaned_patches_sim, dtype=np.float32)
            
            checkpoint_data = {
                'patches_real': patches_real_array,
                'patches_sim': patches_sim_array,
                'processed_count': processed_count,
                'data_min': dataset.data_min,
                'data_max': dataset.data_max,
                'checkpoint_type': 'patches',
                'patch_count': len(cleaned_patches_real),
                'patch_shape': (dataset.patch_size, dataset.patch_size)
            }
            
            temp_checkpoint = f"{checkpoint_path}.tmp"
            torch.save(checkpoint_data, temp_checkpoint)
            os.replace(temp_checkpoint, checkpoint_path)
            
            dataset.logger.debug(f"Patch checkpoint saved: {processed_count} files processed, "
                            f"{len(cleaned_patches_real)} valid patches")
        else:
            dataset.logger.warning(f"No valid patches to save in checkpoint for file {processed_count}")
            
    except Exception as e:
        dataset.logger.error(f"Error saving patch checkpoint: {e}")

def resume_patch_checkpoint(dataset, checkpoint_path):
    """Riprende l'estrazione patch da checkpoint con validazione robusta."""
    try:
        dataset.logger.info("Caricamento checkpoint patch...")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verifica tipo checkpoint
        if checkpoint.get('checkpoint_type') != 'patches':
            dataset.logger.warning("Tipo checkpoint non corretto, ignorato")
            return 0
        
        # Carica patch con gestione errori
        patches_real = checkpoint.get('patches_real', [])
        patches_sim = checkpoint.get('patches_sim', [])
        
        if len(patches_real) > 0 and len(patches_sim) > 0:
            try:
                # Verifica che siano array numpy regolari
                if isinstance(patches_real, np.ndarray) and isinstance(patches_sim, np.ndarray):
                    # Validazione forma
                    expected_shape = (len(patches_real), dataset.patch_size, dataset.patch_size)
                    if (patches_real.shape == expected_shape and 
                        patches_sim.shape == expected_shape):
                        
                        # Conversione sicura in tensori PyTorch
                        dataset.data_real = torch.from_numpy(patches_real.astype(np.float32))
                        dataset.data_sim = torch.from_numpy(patches_sim.astype(np.float32))
                        
                    else:
                        dataset.logger.error(f"Forma checkpoint invalida: real={patches_real.shape}, "
                                        f"sim={patches_sim.shape}, expected={expected_shape}")
                        return 0
                else:
                    # Tentativo di recupero per formato legacy
                    dataset.logger.info("Tentativo recupero checkpoint formato legacy...")
                    
                    # Verifica se sono liste di array
                    if isinstance(patches_real, (list, tuple)) and isinstance(patches_sim, (list, tuple)):
                        valid_patches_real = []
                        valid_patches_sim = []
                        
                        for patch_r, patch_s in zip(patches_real, patches_sim):
                            try:
                                if (isinstance(patch_r, np.ndarray) and isinstance(patch_s, np.ndarray) and
                                    patch_r.shape == (dataset.patch_size, dataset.patch_size) and
                                    patch_s.shape == (dataset.patch_size, dataset.patch_size) and
                                    np.isfinite(patch_r).all() and np.isfinite(patch_s).all()):
                                    
                                    valid_patches_real.append(patch_r.astype(np.float32))
                                    valid_patches_sim.append(patch_s.astype(np.float32))
                            except:
                                continue
                        
                        if len(valid_patches_real) > 0:
                            dataset.data_real = torch.tensor(np.array(valid_patches_real))
                            dataset.data_sim = torch.tensor(np.array(valid_patches_sim))
                            dataset.logger.info(f"Recuperate {len(valid_patches_real)} patch valide da formato legacy")
                        else:
                            dataset.logger.error("Nessuna patch valida nel checkpoint legacy")
                            return 0
                    else:
                        dataset.logger.error(f"Formato checkpoint non riconosciuto: "
                                        f"real={type(patches_real)}, sim={type(patches_sim)}")
                        return 0
                        
            except Exception as e:
                dataset.logger.error(f"Errore nel processing patch da checkpoint: {e}")
                return 0
        
        # Ripristina statistiche
        dataset.data_min = checkpoint.get('data_min', dataset.data_min)
        dataset.data_max = checkpoint.get('data_max', dataset.data_max)
        
        processed_count = checkpoint.get('processed_count', 0)
        patch_count = len(dataset.data_real) if hasattr(dataset.data_real, '__len__') else 0
        
        dataset.logger.info(f"Checkpoint patch caricato: {patch_count} patch ripristinate")
        return processed_count
        
    except Exception as e:
        dataset.logger.error(f"Errore nel caricamento checkpoint patch: {e}")
        # In caso di errore, rimuovi il checkpoint corrotto
        try:
            os.remove(checkpoint_path)
            dataset.logger.info("Checkpoint corrotto rimosso, riavvio da zero")
        except:
            pass
        return 0

def finalize_dataset(dataset, patches_real, patches_sim):
    """Finalizza il dataset convertendo le patch in tensori."""
    if patches_real and patches_sim:
        # Converte in array numpy se necessario
        if isinstance(patches_real[0], list):
            patches_real = [item for sublist in patches_real for item in sublist]
            patches_sim = [item for sublist in patches_sim for item in sublist]
        
        current_real = torch.tensor(np.array(patches_real), dtype=torch.float32)
        current_sim = torch.tensor(np.array(patches_sim), dtype=torch.float32)

        if not isinstance(dataset.data_real, list):
            dataset.data_real = [dataset.data_real] if hasattr(dataset.data_real, 'shape') and dataset.data_real.shape[0] > 0 else []
        if not isinstance(dataset.data_sim, list):
            dataset.data_sim = [dataset.data_sim] if hasattr(dataset.data_sim, 'shape') and dataset.data_sim.shape[0] > 0 else []

        dataset.data_real.append(current_real)
        dataset.data_sim.append(current_sim)

    total_patches = sum(t.shape[0] for t in dataset.data_real) if isinstance(dataset.data_real, list) else len(dataset.data_real)
    dataset.logger.info(f"Dataset finalizzato: {total_patches} patch totali")

def should_free_memory():
    """Determina se è necessario liberare memoria."""
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)  # GB
    return memory_usage > 4.0  # Libera memoria se supera 4GB

def consolidate_and_free_memory(dataset, patches_real, patches_sim):
    """Consolida le patch e libera memoria."""
    if patches_real and patches_sim:
        # Converte in tensori e aggiunge al dataset principale
        current_real = torch.tensor(np.array(patches_real), dtype=torch.float32)
        current_sim = torch.tensor(np.array(patches_sim), dtype=torch.float32)
        
        if not isinstance(dataset.data_real, list):
            dataset.data_real = [dataset.data_real] if len(dataset.data_real) > 0 else []
        if not isinstance(dataset.data_sim, list):
            dataset.data_sim = [dataset.data_sim] if len(dataset.data_sim) > 0 else []

        dataset.data_real.append(current_real)
        dataset.data_sim.append(current_sim)
        
        # Svuota liste e forza garbage collection
        patches_real.clear()
        patches_sim.clear()
        del current_real, current_sim
        gc.collect()
        
        dataset.logger.debug("Memoria liberata e patch consolidate in un nuovo chunk tensoriale")
