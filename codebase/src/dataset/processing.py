import numpy as np
import torch
from scipy.signal import hilbert
from collections import Counter

from scipy.signal import medfilt2d

def apply_median_filter(image_tensor, window_size):
    """
    Apply a median filter to a tensor.

    Args:
        image_tensor (torch.Tensor): The input tensor.
        window_size (int): The size of the median filter window.

    Returns:
        torch.Tensor: The filtered tensor.
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
        
    image_np = image_tensor.squeeze().cpu().numpy()
    filtered_np = medfilt2d(image_np, kernel_size=window_size)
    return torch.from_numpy(filtered_np).unsqueeze(0).unsqueeze(0).to(image_tensor.device)

def to_decibels(image, use_HT=False):
    """
    Convert the image to decibels with validation.

    Args:
        image (numpy.ndarray): The input image.
        use_HT (bool): Flag to indicate whether to use the Hilbert Transform.

    Returns:
        numpy.ndarray: The image in decibels.

    Raises:
        ValueError: If the input contains invalid values.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")
    
    eps = 1e-13
    if use_HT:
        # Apply Hilbert Transform to the image
        image = hilbert(image, axis=1)
    
    # Verifica che l'immagine contenga valori validi
    if not np.isfinite(image).all():
        raise ValueError("Input image contains non-finite values (NaN or Inf)")
    
    # Verifica che ci siano valori positivi dopo aggiunta di eps
    if np.all(image + eps <= 0):
        raise ValueError("All image values are non-positive, cannot convert to dB")
    
    R = 20*np.log10(np.abs(image) + eps)  # Usa valore assoluto per sicurezza
    
    # Verifica il risultato
    if not np.isfinite(R).all():
        raise ValueError("dB conversion produced non-finite values")
    
    return R

def normalize(image, vmin, vmax, normalization_type='range_zero_to_one'):
    # Convert to numpy for easier manipulation
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
    else:
        img_np = image
    
    # Normalize based on the specified type
    if normalization_type == 'range_minus_one_to_one':
        # Normalize to [-1, 1]
        img_norm = 2 * (img_np - vmin) / (vmax - vmin) - 1
    else:
        # Default normalize to [0, 1]
        img_norm = (img_np - vmin) / (vmax - vmin)

    if isinstance(image, torch.Tensor):
        # Convert back to tensor
        return torch.tensor(img_norm, dtype=image.dtype, device=image.device)
    else:
        return img_norm

def get_identifier(path):
    """
    Extract the identifier from the file basename.

    Args:
        path (str): The file path.

    Returns:
        str: The identifier extracted from the file path.
    """
    
    # files name are in the format s_<identifier>_[sim/rgram].[img/xml]
    return path.split("_")[1]

def calculate_box(x, width=256, height=256, R=None):
    """
    Calculate the bounding box around a point in the radargram.

    Args:
        x (int): The x-coordinate of the point.
        width (int): The width of the box.
        height (int): The height of the box.
        R (numpy.ndarray): The radargram data.

    Returns:
        tuple: A tuple containing the coordinates of the bounding box (left_edge, right_edge, top_edge, bottom_edge).
    """
    if R is None:
        raise ValueError("Radargram data R must be provided.")

    M = R.shape[1]
    N = R.shape[0] # Image height

    # Ensure image is large enough for a patch
    if M < width or N < height:
        return None, None, None, None

    left_edge = x - width//2
    right_edge = x + width//2
    
    if left_edge < 0:
        left_edge = 0
        right_edge = width
        x = width//2
    if right_edge > M:
        right_edge = M
        left_edge = M - width
        x = M - width//2
        
    y_current = R[:,x].argmax()  # Get the index of the maximum value in the column (i.e. the y-coordinate of the point)
    horizon_line = R[:,left_edge:right_edge].argmax(axis=0)  # Get the index of the maximum value inside the rectangle
    
    minimum_y = horizon_line.min()
    maximum_y = horizon_line.max()
    
    top_edge = y_current - 100
    bottom_edge = top_edge + height
    
    
    # Box positioning calculation
    free_space = height - (maximum_y - minimum_y)
    threshold = 100  # Define a threshold for free space
    if free_space >= 1:
        # do nothing
        # print("The box is too small")
        available_space = free_space - threshold
        if available_space > 0:
            top_edge = minimum_y - available_space // 3 # assign less than half of the available space above the horizon line
            bottom_edge = top_edge + height

    # Boundary checks
    if top_edge < 0:
        top_edge = 0
        bottom_edge = height
    if bottom_edge > N:
        bottom_edge = N
        top_edge = N - height

    # After all calculations, ensure the box is valid
    if top_edge < 0 or left_edge < 0 or bottom_edge > N or right_edge > M or (bottom_edge - top_edge) != height or (right_edge - left_edge) != width:
         return None, None, None, None
            
    return (left_edge, right_edge, top_edge, bottom_edge), y_current, maximum_y, minimum_y

def extract_patches(R, S, patch_size, patch_overlap):
    """
    Extract patches from the radargram with a predefined width, height, and overlap.

    Args:
        R (numpy.ndarray): The real radargram data.
        S (numpy.ndarray): The simulated radargram data.
    Raises:
        AssertionError: If the shapes of R and S do not match.

    Returns:
        tuple: A tuple containing two lists of patches, one for the real radargram and one for the simulated radargram.
    """
    patches_real = []
    patches_sim = []
    assert R.shape == S.shape, "Real radargram and simulated must have the same shape."
    M = R.shape[1]
    
    size = patch_size

    for x in np.arange(size//2, M-1-size//2, step=size-patch_overlap):
        # Calculate the bounding box around the point x
        box, y_current, maximum_y, minimum_y = calculate_box(x, width=size, height=size, R=R)
        if box is None:
            continue
        left_edge, right_edge, top_edge, bottom_edge = box

        # Extract the patch
        patch_real = R[top_edge:bottom_edge, left_edge:right_edge]
        patch_sim = S[top_edge:bottom_edge, left_edge:right_edge]
        
        if patch_real.shape == (patch_size, patch_size) and patch_sim.shape == (patch_size, patch_size):
            patches_real.append(patch_real)
            patches_sim.append(patch_sim)
        else:
            # This should not happen with the current logic, but as a safeguard
            print(f"Skipping patch with invalid shape: real {patch_real.shape}, sim {patch_sim.shape}")
            
    return patches_real, patches_sim

def handle_irregular_data(raw_data, path, logger):

    """
    Gestisce dati con strutture irregolari convertendoli in array regolari.
    
    Args:
        raw_data: Dati grezzi dal file PDR
        path (str): Path del file per logging
        
    Returns:
        numpy.ndarray: Array regolare bidimensionale
    """
    try:
        # Prova conversione diretta
        if isinstance(raw_data, np.ndarray) and len(raw_data.shape) == 2:
            return raw_data

        # Se è già un array numpy ma con problemi di forma
        if isinstance(raw_data, np.ndarray):
            # Prova a fare reshape se possibile
            if raw_data.size > 0:
                # Trova una forma rettangolare ragionevole
                total_size = raw_data.size
                # Prova alcune forme comuni per radargrammi
                common_shapes = [(3600, -1), (2391, -1), (1800, -1), (900, -1), (450, -1)]
                
                for rows, cols in common_shapes:
                    if cols == -1:
                        if total_size % rows == 0:
                            cols = total_size // rows
                            try:
                                reshaped = raw_data.flatten().reshape(rows, cols)
                                if reshaped.shape[1] > 10:  # Almeno 10 colonne
                                    return reshaped
                            except:
                                continue

        # Se i dati sono una lista di array con lunghezze diverse
        if hasattr(raw_data, '__iter__') and not isinstance(raw_data, str):
            try:
                # Converti in lista di array numpy
                data_list = []
                for item in raw_data:
                    try:
                        if hasattr(item, '__iter__') and not isinstance(item, str):
                            # È un array/lista
                            arr = np.asarray(item, dtype=np.float64)
                            if arr.size > 0:  # Solo se non vuoto
                                data_list.append(arr)
                        elif np.isscalar(item) and np.isfinite(float(item)):
                            # È uno scalare valido
                            data_list.append(np.array([float(item)]))
                    except:
                        continue
                
                if len(data_list) == 0:
                    raise ValueError("Lista dati vuota")
                
                # Analizza le lunghezze
                lengths = [len(row) for row in data_list]
                unique_lengths = list(set(lengths))
                
                logger.debug(f"File {path}: {len(data_list)} righe, lunghezze uniche: {unique_lengths[:10]}")
                
                if len(unique_lengths) == 1:
                   # Tutti della stessa lunghezza, conversione diretta
                    return np.array(data_list)
                else:
                    # Lunghezze diverse, prova strategie multiple
                    
                    # Strategia 1: Usa lunghezza mediana con tolleranza
                    median_length = int(np.median(lengths))
                    tolerance = max(1, int(median_length * 0.05))  # 5% di tolleranza
                    
                    compatible_rows = []
                    for row in data_list:
                        if abs(len(row) - median_length) <= tolerance:
                            if len(row) < median_length:
                                # Padding
                                padded = np.pad(row, (0, median_length - len(row)), 'constant', constant_values=0)
                                compatible_rows.append(padded)
                            elif len(row) > median_length:
                                # Truncate
                                compatible_rows.append(row[:median_length])
                            else:
                                compatible_rows.append(row)
                    
                    if len(compatible_rows) >= len(data_list) * 0.8:  # Almeno 80% delle righe utilizzabili
                        result = np.array(compatible_rows)
                        logger.debug(f"Strategia mediana riuscita per {path}: {result.shape}")
                        return result
                    
                    # Strategia 2: Usa la lunghezza più comune
                    length_counts = Counter(lengths)
                    most_common_length, count = length_counts.most_common(1)[0]
                    
                    if count >= len(data_list) * 0.6:  # Almeno 60% delle righe hanno la stessa lunghezza
                        common_rows = []
                        for row in data_list:
                            if len(row) == most_common_length:
                                common_rows.append(row)
                            elif len(row) > most_common_length:
                                # Truncate alla lunghezza comune
                                common_rows.append(row[:most_common_length])
                            else:
                                # Pad alla lunghezza comune
                                padded = np.pad(row, (0, most_common_length - len(row)), 'constant')
                                common_rows.append(padded)
                        
                        if len(common_rows) > 0:
                            result = np.array(common_rows)
                            logger.debug(f"Strategia lunghezza comune riuscita per {path}: {result.shape}")
                            return result
                    
                    # Strategia 3: Crea matrice sparsa con dimensione massima
                    max_length = max(lengths)
                    if max_length <= 10000:  # Limite ragionevole
                        padded_rows = []
                        for row in data_list:
                            if len(row) < max_length:
                                padded = np.pad(row, (0, max_length - len(row)), 'constant')
                                padded_rows.append(padded)
                            else:
                                padded_rows.append(row[:max_length])
                        
                        if len(padded_rows) > 0:
                            result = np.array(padded_rows)
                            logger.debug(f"Strategia padding massimo riuscita per {path}: {result.shape}")
                            return result
                        
            except Exception as e:
                logger.debug(f"Fallimento gestione lista irregolare per {path}: {e}")
        
        # Ultimo tentativo: estrazione brute force
        try:
            logger.debug(f"Tentativo estrazione brute force per {path}")
            
            # Estrai tutti i numeri validi dalla struttura
            def extract_numbers(obj):
                numbers = []
                if np.isscalar(obj):
                    try:
                        if np.isfinite(float(obj)):
                            numbers.append(float(obj))
                    except:
                        pass
                elif hasattr(obj, '__iter__') and not isinstance(obj, str):
                    for item in obj:
                        numbers.extend(extract_numbers(item))
                return numbers
            
            all_numbers = extract_numbers(raw_data)
            
            if len(all_numbers) >= 1000:  # Almeno 1000 punti dati
                # Crea una griglia ragionevole
                total_points = len(all_numbers)
                
                # Prova forme comuni, preferendo quelle più vicine al quadrato
                candidate_shapes = []
                for rows in [50, 100, 200, 300, 450, 600, 900, 1200, 1800, 2400, 3600]:
                     if total_points >= rows * 10:  # Almeno 10 colonne
                        cols = total_points // rows
                        candidate_shapes.append((rows, cols, abs(rows - cols)))  # preferenza forme quadrate
                
                # Ordina per preferenza (più vicino al quadrato)
                candidate_shapes.sort(key=lambda x: x[2])
                
                for rows, cols, _ in candidate_shapes[:3]:  # Prova le 3 migliori
                    try:
                        usable_points = rows * cols
                        if usable_points <= total_points:
                            data_subset = all_numbers[:usable_points]
                            result = np.array(data_subset).reshape(rows, cols)
                            logger.debug(f"Estrazione brute force riuscita per {path}: {result.shape}")
                            return result
                    except:
                        continue
                        
        except Exception as e:
            logger.debug(f"Fallimento estrazione brute force per {path}: {e}")
        
        raise ValueError(f"Tutte le strategie di recupero fallite per struttura irregolare")
        
    except Exception as e:
        raise ValueError(f"Errore nella gestione dati irregolari: {str(e)}") from e
