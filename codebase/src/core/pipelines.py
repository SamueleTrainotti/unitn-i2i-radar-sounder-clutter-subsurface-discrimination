import torch
from core.config import get_config
from dataset import MapDataset
from core.anomaly_detector import AnomalyDetector

def run_anomaly_detection_pipeline(limit_to_sample=None):
    """
    Detects anomalies in radargrams using a trained generator model.
    The configuration is expected to be globally set.

    Args:
        limit_to_sample (int, optional): If provided, the detection will only
                                         run on the sample at this index in the dataset.
    """
    config = get_config()

    # --- Anomaly Detection ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["MODEL_NAME"] == "pix2pix":
        from models import Pix2Pix
        model = Pix2Pix(training=False)
    elif config["MODEL_NAME"] == "cyclegan":
        from models import CycleGAN
        model = CycleGAN(training=False)
    else:
        raise ValueError("MODEL_NAME must be 'pix2pix' or 'cyclegan'.")

    model.load(config['MODEL_PATH'])
    generator = model.get_generator()

    # Create dataset using the merged config
    full_dataset = MapDataset(
        root_dir=config['INPUT_DATA']['TEST'],
        temp_dir=config["OUTPUT_DATA"]["TEMP"],
        temp_dataset=config["OUTPUT_DATA"]["TEMP_DATASET"],
        SAVE_DATASET=config.get("SAVE_DATASET", False),
        LOAD_DATASET=config.get("LOAD_DATASET", False),
        patch_size=config.get("PATCH_SIZE", 256),
        patch_overlap=config.get("PATCH_OVERLAP", 128),
        normalization_type=config.get("NORMALIZATION_TYPE", "range_zero_to_one"),
        augmentation_config=config.get("AUGMENTATION"),
        normalization_config=config.get("NORMALIZATION"),
    )

    if limit_to_sample is not None and limit_to_sample < len(full_dataset):
        dataset = torch.utils.data.Subset(full_dataset, [limit_to_sample])
    else:
        dataset = full_dataset
    
    output_dir = config['OUTPUT_DATA']['EVALUATION']
    threshold = config.get('THRESHOLD', 0.5)
    
    anomaly_detector = AnomalyDetector(generator, device)
    anomaly_detector.detect_anomalies(dataset, output_dir, threshold, sample_index=limit_to_sample)
